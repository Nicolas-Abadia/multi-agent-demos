import json
from typing import Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv
import textwrap

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# Initialize the LLM
# Initialize different LLMs with their own system messages

search_query_llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0.7,
)

researcher_llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0.7,
)

recommend_llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
)

# Initialize Tavily search tool - explain how the agents can use numerous different tools
search = TavilySearchResults(max_results=7)

# Define the state schema - keeps track of the conversation history
class AppState(TypedDict):
    messages: list[BaseMessage]

# Add a simple formatting helper function
def format_output(title, content, width=80):
    """Format output to be more concise and fit in terminal"""
    separator = "-" * min(width, len(title) + 4)
    print(f"\n{separator}\n| {title} |")
    print(separator)
    
    if isinstance(content, str):
        # Wrap text to fit terminal width
        wrapped = textwrap.fill(content, width=width)
        print(wrapped)
    else:
        print(content)

# Define the tool-calling node
def search_tool(state: Dict) -> Dict:
    """
    Refines the user's vacation preferences into a specific search query and retrieves search results.

    Args:
        state (Dict): The current state of the conversation, including messages.

    Returns:
        Dict: The updated state with the search results appended to the messages.
    """
    messages = state["messages"]
    # First use LLM to refine the search query
    refine_messages = [
        SystemMessage(content='''
            You are a search query specialist. Your task is to reformulate user 
            vacation preferences into specific, targeted search terms that will yield the most relevant vacation destinations.
            Return only the search query, nothing else.'''),

        HumanMessage(content=f"Convert this request into a specific internet search query: {messages[-1].content}")
    ]

    refined_query = search_query_llm.invoke(refine_messages).content
    format_output("SEARCH QUERY", refined_query)
    
    # Use refined query with Tavily
    search_results = search.invoke({
        "query": messages[-1].content,
        "search_depth": "basic"
    })
    
    # Store full results in message
    formatted_results = json.dumps(search_results, indent=2)
    messages.append(AIMessage(content=(formatted_results)))
    
    # Only display a preview of search results
    preview = []
    if search_results:
        for i, result in enumerate(search_results):
            url = result.get("url", "No URL")
            content = result.get("content", "")

            preview.append(f"Result {i+1}: {url}\n{content}")
    
    format_output("SEARCH RESULTS", 
                 f"{len(search_results)} results found.\n\n" + "\n\n".join(preview))
    
    return {"messages": messages}

# Define the research node
def research(state: Dict) -> Dict:
    """
    Analyzes search results and extracts key findings about vacation destinations.

    Args:
        state (Dict): The current state of the conversation, including messages.

    Returns:
        Dict: The updated state with the research findings appended to the messages.
    """
    messages = state["messages"]
    research_messages = [
        SystemMessage(content='''
            You are an expert vacation planner and research analyst. Analyze the provided search results and extract key vacation 
            destinations along with detailed information about what they offer. 
            
            For each destination, please include:
            - **Destination Name**: The name or location of the vacation spot.
            - **Key Features**: Unique attractions or benefits (e.g., scenic views, cultural sites, adventure activities).
            - **Amenities & Activities**: Information on accommodations, dining, recreational activities, and local experiences.
            - **Pros & Cons**: Brief evaluation points that can help in deciding if the destination fits various user preferences.
            - **Actionable Tips**: Recommendations for planning a visit (e.g., best time to visit, must-see attractions, local travel tips).
            
            Organize your response into clear sections for each destination. Use bullet points or headings where appropriate for clarity.
        '''),
        *messages
    ]
    response = researcher_llm.invoke(research_messages)
    format_output("RESEARCH ANALYSIS", response.content)
    return {"messages": messages + [response]}

# Define the explain node
def recommend(state: Dict) -> Dict:
    """
    Provides professional recommendations for vacation destinations based on research findings.

    Args:
        state (Dict): The current state of the conversation, including messages.

    Returns:
        Dict: The updated state with the recommendations appended to the messages.
    """
    messages = state["messages"]
    recommend_messages = [
        SystemMessage(content='''
            You are an expert vacation planner known for providing clear and professional recommendations.
            
            Based on the research findings provided, please perform the following tasks:
            1. Identify and rank the top vacation destinations that best meet the user's query.
            2. For each top destination, provide:
                - **Destination Name**: The vacation spot's name or location.
                - **Information**: A detailed explanation of why this destination is ideal, including unique features, amenities, and any standout attractions.
                - **Recommendations**: Actionable tips or suggestions for planning a visit, such as the best time to travel, local must-see attractions, and any insider advice.
                
            Organize your response in a clear, structured format (using headings or bullet points) to ensure it is easy to understand.
            Be as detailed and informative as necessary, as a travel agent would be when providing a recommendation to a client.
        '''),
        *messages
    ]
    response = recommend_llm.invoke(recommend_messages)
    format_output("RECOMMENDATIONS", response.content)
    return {"messages": messages + [response]}

# Build the graph
graph = StateGraph(AppState)
graph.add_node("search", search_tool)
graph.add_node("research", research)
graph.add_node("recommend", recommend)

# Define the edges
graph.set_entry_point("search")
graph.add_edge("search", "research")
graph.add_edge("research", "recommend")
graph.add_edge("recommend", END)

# Compile the graph
app = graph.compile()

# Define your function to run the graph
def run_conversation(user_input: str):
    """
    Runs the conversation graph with the given user input.

    Args:
        user_input (str): The user's input message.

    Returns:
        str: The final output message from the conversation.
    """
    format_output("USER QUERY", user_input)
    
    initial_state = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }
    output = app.invoke(initial_state)
    
    final_response = output["messages"][-1].content
    format_output("FINAL RESULTS", final_response)
    return final_response

if __name__ == "__main__":
    result = run_conversation("Im looking for a relaxing spa vacation in a bleak desert environment")
    print(result)