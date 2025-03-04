'''
EXAMPLE OF A SUPERVISOR AGENT WORKFLOW
GOAL: Create a workflow that can handle a golf-related query and route it to the appropriate worker
WORKFLOW: 1. Tavily search for golf tips
          2. Researcher analyzes search results and extracts relevant golf tips
          3. Supervisor routes to the next worker
          4. Drill creator formats and structures the information into clear, actionable steps
'''

import os
from dotenv import load_dotenv
from typing import Dict, Literal, List, TypedDict, Union
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import logging
import json

# Setup improved logging with colors and formatting
class ColorFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[94m',  # Blue
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'DEBUG': '\033[92m',  # Green
        'RESET': '\033[0m'  # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

# Configure logging with color
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Load environment variables
load_dotenv()
if not os.getenv("TAVILY_API_KEY"):
    logger.error("TAVILY_API_KEY not found in environment variables")
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment variables")

# Define state types consistently
class State(TypedDict):
    """State object for the workflow."""
    messages: List[BaseMessage]
    next: str

# Define router type
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["tavily_search", "researcher", "drill_creator", "FINISH"]

# Initialize LLMs and tools
supervisor_llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
researcher_llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
drill_creator_llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
search_query_llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

# Initialize Tavily search tool with proper configuration
tavily_tool = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    search_depth="advanced",  # Changed to advanced from basic
    include_raw_content=True,
    include_domains=["golfdigest.com", "golf.com", "pgatour.com", "golfchannel.com"],  # Add golf-specific domains
    exclude_domains=["youtube.com", "tiktok.com", "reddit.com"]  # Exclude social media
)

def print_section(title, content, symbol="="):
    """Helper function to print formatted sections"""
    border = symbol * 50
    print(f"\n\033[1;36m{border}\n{title}\n{border}\033[0m")
    print(content)

def tavily_search(state: State) -> State:
    """Execute Tavily search based on the last message."""
    messages = state["messages"]
    
    # Get the user query - check if it's from the beginning or from a message
    user_query = messages[0].content if messages else "How to fix a golf slice"
    
    # First, directly search for specific golf slice fixes - don't reformulate yet
    try:
        direct_search_results = tavily_tool.invoke({
            "query": f"golf slice correction drills and techniques {user_query}",
            "search_depth": "advanced" 
        })
        
        if direct_search_results and len(direct_search_results) >= 2:
            # We got good results, use them
            formatted_results = json.dumps(direct_search_results, indent=2)
            print_section("DIRECT SEARCH RESULTS", formatted_results)
            messages.append(AIMessage(content=formatted_results))
            return {'messages': messages, 'next': state['next']}
    except Exception as e:
        print(f"Direct search failed: {e}")
    
    # If direct search didn't work well, try with query refinement
    system_message = [
        SystemMessage(content='''
            You are a search query specialist for golf improvement.
            
            Create a specific search query that will find actionable golf drills and techniques
            to fix slicing the ball. Focus on:
            - Professional golf instruction techniques
            - Specific drills to fix a golf slice
            - Technical corrections for golf swing path
            
            Return ONLY the search query text, nothing else. Make it specific and detailed.
        '''),
        HumanMessage(content=f'Create a search query to find specific golf drills for fixing: {user_query}')
    ]
    
    refined_query = search_query_llm.invoke(system_message).content
    print_section("SEARCH QUERY", refined_query, "-")
    
    try:
        # Use the refined query with additional parameters
        search_results = tavily_tool.invoke({
            "query": refined_query,
            "search_depth": "advanced",
            "k": 10  # Request more results
        })
        
        # If no results or empty results, try a more general query
        if not search_results or len(search_results) < 2:
            search_results = tavily_tool.invoke({
                "query": "how to fix a golf slice drills techniques",
                "search_depth": "advanced"
            })
        
        formatted_results = json.dumps(search_results, indent=2)
        print_section("SEARCH RESULTS", formatted_results)
        messages.append(AIMessage(content=formatted_results))
        return {'messages': messages, 'next': state['next']}
        
    except Exception as e:
        error_message = f"Search error: {str(e)}"
        logger.error(error_message)

def supervisor_node(state: State) -> Command:
    """Route to the next appropriate worker."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Check if we need to initialize with search
    if len(messages) <= 1:
        print_section("SUPERVISOR DECISION", "Initiating search")
        return Command(goto="tavily_search", update={"next": "tavily_search"})
    
    # Check if we have search results but no analysis
    try:
        # Try to parse the last message as JSON to see if it's search results
        json_data = json.loads(last_message)
        if isinstance(json_data, list) and len(json_data) > 0:
            print_section("SUPERVISOR DECISION", "Routing to researcher for analysis")
            return Command(goto="researcher", update={"next": "researcher"})
    except:
        # Not JSON, could be analysis ready for drill creation
        pass
    
    system_prompt = f'''
        You are a supervisor coordinating a golf instruction team. Determine which worker should handle the next step:
        
        1. tavily_search: Conducts web search for golf tips (use if we need more or better search results)
        2. researcher: Analyzes search results to extract key golf tips (use if we have raw search results)
        3. drill_creator: Creates step-by-step instructions from analyzed information (use if we have analyzed information)
        4. FINISH: Complete the workflow (use if we have clear, actionable instructions for the golfer)
        
        Last message snippet: "{last_message[:200]}..."
        
        Respond ONLY with: "tavily_search", "researcher", "drill_creator", or "FINISH".
    '''
    
    formatted_messages = [SystemMessage(content=system_prompt)]
    formatted_messages.extend(messages)
    
    response = supervisor_llm.with_structured_output(
        Router,
        method="function_calling"  # Fix the structured output method
    ).invoke(formatted_messages)
    
    print_section("SUPERVISOR DECISION", f"Routing to: {response['next']}")
    goto = response["next"]
    
    if goto == "FINISH":
        goto = END
        
    return Command(goto=goto, update={"next": goto})

def researcher_node(state: State) -> State:
    """Analyze search results and extract relevant golf tips."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    system_prompt = '''
        You are a golf improvement researcher specializing in analyzing information to help golfers.
        
        Your task:
        1. Analyze the search results provided
        2. Extract the most relevant and practical golf tips, techniques, and drills
        3. Organize the information into clear categories
        4. Focus on actionable advice that addresses the user's specific golf concerns
        5. Cite your sources where relevant
        
        Structure your response with clear headings and bullet points for readability.
        Prioritize quality tips that golfers can actually implement in their practice.
    '''

    researcher_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze these search results and extract key golf tips: {last_message}")
    ]

    response = researcher_llm.invoke(researcher_messages)
    print_section("RESEARCHER ANALYSIS", response.content)
    messages.append(AIMessage(content=response.content))
    return {"messages": messages, "next": state["next"]}

def drill_creator_node(state: State) -> State:
    """Format and structure the analyzed information."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    system_prompt = '''
        You are a golf instruction expert specializing in creating clear, structured practice routines.
        
        Your task:
        1. Take the analyzed golf tips and information
        2. Transform them into a clear, step-by-step drills
        4. Include specific instructions on:
           - Proper setup and positioning
           - Step-by-step execution (important for drills)
           - Common mistakes to avoid
           - How to track improvement
        
        Format your response with:
        - Clear headings and numbered steps
        - Bullet points for key tips
        
        Make your instructions practical and actionable for a golfer to follow without additional guidance.
    '''

    drill_creator_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Format these golf tips into structured practice drills: {last_message}")
    ]

    response = drill_creator_llm.invoke(drill_creator_messages)
    print_section("DRILL CREATOR OUTPUT", response.content)
    messages.append(AIMessage(content=response.content))
    return {"messages": messages, "next": state["next"]}

# Create and configure the workflow graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("tavily_search", tavily_search)
workflow.add_node("researcher", researcher_node)
workflow.add_node("drill_creator", drill_creator_node)
workflow.add_node("supervisor", supervisor_node)

# Add edges

workflow.add_edge("tavily_search", "supervisor")
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("drill_creator", "supervisor")
workflow.set_entry_point("supervisor")

# Compile the graph
graph = workflow.compile()

def process_golf_query(query: str) -> List[BaseMessage]:
    """Process a golf-related query through the workflow."""
    state = {
        'messages': [HumanMessage(content=query)],
        'next': ''
    }
    print_section("INITIAL QUERY", query)
    final = graph.invoke(state)
    return final["messages"]

# Example usage updating for better testing
if __name__ == "__main__":
    # Test with a more specific golf slice question
    result = process_golf_query("How do I fix my driver slice? I always slice the ball to the right.")
    
    print_section("FINAL RESULT", "=" * 70, "=")
    for i, message in enumerate(result):
        if i == 0:  # Skip the initial query
            continue
        if i == len(result) - 1:  # Format the final answer specially
            print("\n\033[1;32m" + "=" * 70 + "\nFINAL ANSWER\n" + "=" * 70 + "\033[0m")
            print(f"\n{message.content}\n")
        else:
            prefix = f"[{message.type}]:"
            print(f"\n{prefix}\n{'-' * len(prefix)}\n{message.content[:200]}...\n")
