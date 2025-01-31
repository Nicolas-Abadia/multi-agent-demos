
import json
from typing import Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize the LLM
# Initialize different LLMs with their own system messages

researcher_llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0.7,
)

explainer_llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
)
# Initialize Tavily search tool
tavily_tool = TavilySearchResults(
    max_results=6,
    search_depth="advanced",
    include_raw_content=True,
    include_domains=[],  # Example domains to include
    exclude_domains=["youtube.com", "tiktok.com", "reddit.com"],  # Example domains to exclude
    k=10
)# Define the tool-calling node
def call_tool(state: Dict) -> Dict:
    messages = state["messages"]

    search_llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0.3
    )
    # First use LLM to refine the search query
    refine_messages = [
        SystemMessage(content='''
            You are a search query specialist. Your task is to reformulate user 
            golf queries into specific, targeted search terms that will yield the most relevant golf tips and drills.
            Do NOT use tiktok or youtube in the search query or as results. Find specific golf drills 
            for the respective problem. Look into drills.
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
    print(f'\n====TOOL NODE=====\n{formatted_results}')
    return {"messages": messages}

# Define the research node
def research(state: Dict) -> Dict:
    messages = state["messages"]
    research_messages = [
        SystemMessage(content='''
            You are an expert golf research analyst. Analyze the search results and 
            extract key findings, methodologies, and important conclusions about fixing 
            the golfer's specific problem. Focus on actionable drills and fixes.
            Organize your response into clear sections with specific tips and drills.'''),
        *messages
    ]
    response = researcher_llm.invoke(research_messages)
    print(f'\n=====RESEARCH NODE=====\n{response}')
    return {"messages": messages + [response]}

# Define the explain node
def explain(state: Dict) -> Dict:
    messages = state["messages"]
    explain_messages = [
        SystemMessage(content='''
            You are an expert golf instructor known for clear, practical explanations.
            Take the research findings and explain them in simple, actionable steps.
            Include:
            1. A clear explanation of what's causing the problem
            2. 3-4 specific drills to fix it
            3. Common mistakes to avoid
            Use simple analogies when helpful and focus on practical advice.'''),
        *messages
    ]
    response = explainer_llm.invoke(explain_messages)
    print(f"\n=====EXPLAIN NODE====={response}")
    return {"messages": messages + [response]}

# Define routing logic
def should_call_tool(state: Dict) -> str:
    query = state["messages"][-1].content.lower()
    return "tool" if "search" in query else "research"

# Define the state schema
class AppState(TypedDict):
    messages: list[BaseMessage]

# Build the graph
graph = StateGraph(AppState)
graph.add_node("tool", call_tool)
graph.add_node("research", research)
graph.add_node("explain", explain)

# Define the edges
graph.add_edge("tool", "research")
graph.add_edge("research", "explain")
graph.add_edge("explain", END)
graph.set_entry_point("tool")


# Compile the graph
app = graph.compile()

# Define your function to run the graph
def run_conversation(user_input: str):
    initial_state = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }
    output = app.invoke(initial_state)
    return output["messages"][-1].content

if __name__ == "__main__":
    result = run_conversation("I keep topping the ball")
    print(result)