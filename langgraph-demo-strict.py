import json
from typing import Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# Initialize the LLM
# Initialize different LLMs with their own system messages

search_llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0.3
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
tavily_tool = TavilySearchResults(
    max_results=6,
    search_depth="advanced",
    include_raw_content=True,
    include_domains=[],  # Example domains to include
    exclude_domains=["youtube.com", "tiktok.com", "reddit.com"],  # Example domains to exclude
    k=10

)

# Define the state schema - keeps track of the conversation history
class AppState(TypedDict):
    messages: list[BaseMessage]

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
            Do NOT use tiktok or youtube in the search query or as results. Find specific destinations 
            for the respective problem. Look into lower and higher end locations.
            Return only the search query, nothing else.'''),

        HumanMessage(content=f"Convert this request into a specific search query: {messages[-1].content}")
    ]
    refined_query = search_llm.invoke(refine_messages).content
    
    # Use refined query with Tavily
    search_results = tavily_tool.invoke(
        {"query": refined_query,
        "max_tokens": 3000}
    )
    formatted_results = json.dumps(search_results, indent=2)
    messages.append(AIMessage(content=json.dumps(formatted_results)))
    print(f'\n====SEARCH TOOL NODE=====\n{formatted_results}')
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
            You are an expert vacation planner and research analyst. Analyze the search results and 
            extract key findings, methodologies, and important conclusions about fixing 
            the golfer's specific problem. Focus on actionable drills and fixes.
            Organize your response into clear sections with specific tips and drills.'''),
        *messages
    ]
    response = researcher_llm.invoke(research_messages)
    print(f'\n=====RESEARCH NODE=====\n{response}')
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
            You are an expert vacation planner known for clear professional recommendations.
            Take the research findings recommend top vacation destinations.
            Include:
            1. A list of the top vacation destinations meeting the user query
            2. What makes these destinations ideal for the user's needs
            3. Include amenities, activities, and unique features of each destination
            .'''),
        *messages
    ]
    response = recommend_llm.invoke(recommend_messages)
    print(f"\n=====EXPLAIN NODE====={response}")
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
    initial_state = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }
    output = app.invoke(initial_state)
    return output["messages"][-1].content

if __name__ == "__main__":
    result = run_conversation("Im looking for a relaxing spa vacation in a bleak desert environment")
    print(result)