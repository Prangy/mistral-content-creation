from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from duckduckgo_search import DDGS
from langchain.agents import initialize_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from serpapi import GoogleSearch
from langchain.agents import create_react_agent

import os
os.environ['SERPAPI_API_KEY'] ="**************************************"

os.environ['TAVILY_API_KEY'] ="tvly-***********************************"
tavily_tool = TavilySearchResults()

# Set up the LLM which will power our application.
model = Ollama(model='mistral')

#Blog Search tool from search the blog urls
@tool
def blog_search_tool(query: str):
    """
    This function interacts with the Google Search API to retrieve search results based on the provided query.

    Args:
        query (str): The search query.
        api_key (str): The API key for accessing the Google Search API.

    Returns:
        dict: The search results obtained from the Google Search API.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": "**********************************************"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results


@tool
def converse(input: str) -> str:
    "Provide a natural language response using the user input."
    return model.invoke(input)

@tool
def generate_content(input: dict) -> str:
    """
    This function generates content for the ContentWriter agent based on the provided input data.

    Args:
        input_data (dict): Input data containing outputs from the web searcher and blog searcher.

    Returns:
        str: Generated content for the ContentWriter agent.
    """
    return model.invoke(input)
    
tools = [blog_search_tool, tavily_tool]

# Initialize agents
agents = {
    "WebSearcher_agent": "zero-shot-react-description",
    "WebSearcherQuality_agent": "zero-shot-react-description",
    "BlogSearcher_agent": "zero-shot-react-description",
    "BlogSearcherQuality_agent": "zero-shot-react-description",
    "ContentWriter_agent": "zero-shot-react-description"
}

initialized_agents = {}
for agent_name, agent_type in agents.items():
    initialized_agents[agent_name] = initialize_agent(
        agent=agent_type,
        tools=[blog_search_tool, tavily_tool],
        llm=model,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True  # Add this line
    )
# Set up message history.
msgs = ChatMessageHistory()
if len(msgs.messages) == 0:
    msgs.add_ai_message("I can search the internet or just chat! How can I help you?")

# Define function to get agent prompt
def get_agent_prompt(agent_name):
    prompts = {
        "web_search_prompt": "You are a web searcher. Search the internet for information.",
        "web_Searcher_Quality_system_prompt": "You are a web searcher Quality Agent. Analysis the web searcher agent work and provide the relevance, grammatical correctness, engagement, harmfulness, and helpfulness on a scale of 1 to 100 in a tabular form of the provided content. If the score is below 90, then ask the web searcher to generate again.",
        "blog_search_prompt": """You are a blog searcher. Your task is to utilize the Google Search API to search for blogs or articles based on the provided information. Your goal is to find relevant blog posts or articles that match the input criteria and return the URL of the blog or article in the final output. Make sure to leverage the capabilities of the Google Search API to retrieve high-quality and relevant content.""",
        "blog_searcher_quality_system_prompt": """You are a blog searcher quality agent. Your task is to assess the quality of the blog searcher agent for each provided blog or article URL. Analyze the content and provide ratings on a scale of 1 to 100 for relevance, grammatical correctness, engagement, harmfulness, and helpfulness. Present your evaluations in a tabular form, detailing the scores for each aspect alongside the corresponding URL. If the score is below 90, then ask the blog searcher to generate again.""",
        "content_writer_system_prompt": """As a content writer, your role is to harness the outputs from the web searcher and blog searcher to craft compelling and impactful content. Integrating information from various sources, your task is to generate engaging narratives enriched with relevant data and imagery. Ensure that the content you produce is both interesting and effective, utilizing the links of the blogs and images provided. Consolidate the gathered data into coherent articles, weaving together insights from the web, captivating imagery, and informative blog posts. At the conclusion of your writing, include the URLs of the articles and images, aligning each image URL with its appropriate context within the content."""
    }
    return prompts.get(agent_name, " How can I assist you today?")

# Define agent names and corresponding prompts
agent_prompts = {
    "WebSearcher_agent": "web_search_prompt",
    "WebSearcherQuality_agent": "web_Searcher_Quality_system_prompt",
    "BlogSearcher_agent": "blog_search_prompt",
    "BlogSearcherQuality_agent": "blog_searcher_quality_system_prompt",
    "ContentWriter_agent": "content_writer_system_prompt"
}

# Create prompts from agent names
agent_messages = [("system", get_agent_prompt(agent_prompts[agent_name])) for agent_name in agents.keys()]

# Create prompt template
prompt = ChatPromptTemplate.from_messages(agent_messages + [("user", "{input}")])

# Define a function which returns the chosen tools as a runnable, based on user input.
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

# The main chain: an LLM with tools.
chain = prompt | model | JsonOutputParser() | tool_chain

# Main chat

user_input = input("User Question: ")
        
# WebSearcher agent's turn
web_searcher_response = initialized_agents["WebSearcher_agent"].invoke(user_input)
print("WebSearcher agent:", web_searcher_response)
        
# Review by WebSearcherQuality agent
web_searcher_quality_prompt = get_agent_prompt("WebSearcherQuality_agent")
web_searcher_quality_response = initialized_agents["WebSearcherQuality_agent"].invoke(web_searcher_response)
print("WebSearcher Quality Agent:", web_searcher_quality_prompt)
print("WebSearcher Quality Agent response:", web_searcher_quality_response)
        
# BlogSearcher agent's turn
blog_searcher_response = initialized_agents["BlogSearcher_agent"].invoke(web_searcher_response)
print("BlogSearcher agent:", blog_searcher_response)
            
# Review by BlogSearcherQuality agent
blog_searcher_quality_prompt = get_agent_prompt("BlogSearcherQuality_agent")
blog_searcher_quality_response = initialized_agents["BlogSearcherQuality_agent"].invoke(blog_searcher_response)
print("BlogSearcher Quality Agent:", blog_searcher_quality_prompt)
print("BlogSearcher Quality Agent response:", blog_searcher_quality_response)
            
# ContentWriter agent's turn
content_writer_prompt = get_agent_prompt("ContentWriter_agent")
content_writer_response = initialized_agents["ContentWriter_agent"].invoke(blog_searcher_response)
print("ContentWriter agent:", content_writer_prompt)
print("ContentWriter agent response:", content_writer_response)
  
