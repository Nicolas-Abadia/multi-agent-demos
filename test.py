from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json

# Load environment variables (make sure you have OPENAI_API_KEY and TAVILY_API_KEY in .env)
load_dotenv()

# Initialize our models and tools
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
search = TavilySearchResults(max_results=5)

def research_topic(query: str) -> str:
    """
    Research a topic using Tavily search and analyze results with GPT-4
    """
    # First, perform the search
    search_results = search.invoke({
        "query": query,
        "search_depth": "advanced"
    })
    
    # Format search results
    formatted_results = json.dumps(search_results, indent=2)
    print(formatted_results)
    
    # Analyze results with GPT-4
    messages = [
        SystemMessage(content="""You are a helpful research assistant. 
        Analyze the search results and provide a comprehensive summary 
        with key insights. Include relevant sources."""),
        HumanMessage(content=f"Please analyze these search results about: {query}\n\nResults: {formatted_results}")
    ]
    
    response = llm.invoke(messages)
    return response.content

def main():
    while True:
        user_query = input("\nWhat would you like to research? (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
            
        print("\nResearching your topic...\n")
        result = research_topic(user_query)
        print("\nAnalysis Results:")
        print("-" * 50)
        print(result)
        print("-" * 50)

if __name__ == "__main__":
    main()