'''

EXAMPLE OF A SUPERVISOR AGENT WORKFLOW
GOAL: Create a workflow that can handle a golf-related query and route it to the appropriate worker
WORKFLOW: 1. Tavily search for golf tips
          2. Researcher analyzes search results and extracts relevant golf tips
          3. Supervisor routes to the next worker
          4. Drill creator formats and structures the information into clear, actionable steps

'''




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
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Define state types consistently
class State(TypedDict):
    """State object for the workflow."""
    messages: List[BaseMessage]
    next: str

# Define router type
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["researcher", "drill_creator", "FINISH"]

# Initialize LLMs and tools
supervisor_llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
researcher_llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
explainer_llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
tavily_llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
tavily_tool = TavilySearchResults(max_results=5)

def tavily_search(state: State) -> State:
    """Execute Tavily search based on the last message."""
    messages = state["messages"]
    system_message = [
        SystemMessage(content='''You are a search query specialist. Your task is to reformulate user 
        golf queries into specific, targeted search terms that will yield the most relevant golf tips and drills.
        Do NOT use tiktok or youtube in the search query or as results. Find specific golf drills 
        for the respective problem. Look into drills. Return only the search query, nothing else.'''),
        HumanMessage(content=f'Convert this request into a specific search query: {messages[-1].content}')
    ]
    
    refined_query = tavily_llm.invoke(system_message).content
    search_results = tavily_tool.invoke({
        "query": refined_query,
        "max_tokens": 100
    })
    
    formatted_results = json.dumps(search_results, indent=2)
    print(formatted_results)
    messages.append(AIMessage(content=formatted_results))
    return {'messages': messages, 'next': state['next']}

def supervisor_node(state: State) -> Command:
    """Route to the next appropriate worker."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    system_prompt = f"""You are a supervisor managing a team of specialized workers for golf improvement:
    
    - researcher: Analyzes search results and extracts relevant golf tips and drills
    - drill_creator: Formats and structures the information into clear, actionable steps
    
    Last message in the conversation: "{last_message}"
    
    Examine this last message and determine which worker should act next.
    Respond ONLY with: "researcher", "drill_creator", or "FINISH"."""
    
    formatted_messages = [SystemMessage(content=system_prompt)]
    formatted_messages.extend(messages)
    
    response = supervisor_llm.with_structured_output(Router).invoke(formatted_messages)
    print(response)
    goto = response["next"]
    
    if goto == "FINISH":
        goto = END
        
    return Command(goto=goto, update={"next": goto})

def researcher_node(state: State) -> State:
    """Analyze search results and extract relevant golf tips."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    system_prompt = """You are a golf improvement researcher. Analyze the search results and extract
    the most relevant and practical golf tips and drills to create helpful content for golfers. Cite your sources"""

    researcher_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze these search results and extract key golf tips: {last_message}")
    ]

    response = researcher_llm.invoke(researcher_messages)
    print(response)
    messages.append(AIMessage(content=response.content))
    return {"messages": messages, "next": state["next"]}

# def coder_node(state: State) -> State:
#     """Format and structure the analyzed information."""
#     messages = state["messages"]
#     last_message = messages[-1].content if messages else ""

#     system_prompt = """You are a golf instruction formatter. Structure the analyzed tips into
#     clear, step-by-step instructions with proper markdown formatting."""

#     coder_messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=f"Format these golf tips into structured instructions: {last_message}")
#     ]

#     response = explainer_llm.invoke(coder_messages)
#     messages.append(AIMessage(content=response.content))
#     return {"messages": messages, "next": state["next"]}

# Create and configure the workflow graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("researcher", tavily_search)
workflow.add_node("drill_creator", researcher_node)
workflow.add_node("supervisor", supervisor_node)

# Add edges
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
    final = graph.invoke(state)
    return final["messages"]

# Example usage
if __name__ == "__main__":
    result = process_golf_query("How to improve my golf swing?")
    for message in result:
        print(f"\n{message.type}: {message.content}\n")
