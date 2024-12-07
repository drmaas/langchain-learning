from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from pydantic import BaseModel, Field

from typing import Annotated, Literal, TypedDict

import asyncio
import os
import requests

from langchain_community.utilities import OpenWeatherMapAPIWrapper

OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

def get_response(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        print("Error:", response.status_code)
    return data

class GetGeoData(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    
class GetWeather(BaseModel):
    lat: int = Field(..., description="The latitude")  
    lon: int = Field(..., description="The longitude")       
    
@tool("get_geo_data_from_city", args_schema=GetGeoData, return_direct=False)
def get_geo_data_from_city(location: str) -> str:
    '''Get geo data from the given city'''
    country_code=840
    url = f'http://api.openweathermap.org/geo/1.0/direct?q={location},{country_code}&limit=1&appid={OPENWEATHERMAP_API_KEY}'  
    geo_data = get_response(url)[0]
    return { 'lat': int(geo_data['lat']), 'lon': int(geo_data['lon']) }
    
@tool("get_current_weather", args_schema=GetWeather, return_direct=True)
def get_current_weather(lat: int, lon: int) -> str:
    '''Get the current weather from geo lat and long'''
    url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=imperial' 
    data = get_response(url)
    return data["list"][0]
        
# tools
tools = [get_geo_data_from_city, get_current_weather]  
  
def createModel():
    # llm = HuggingFaceEndpoint(
    #     repo_id="HuggingFaceH4/zephyr-7b-beta"
    # ) 
    # return ChatHuggingFace(llm=llm, verbose=True).bind_tools(tools)
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=1,
        verbose=True,
    ).bind_tools(tools)

model = createModel()

# prompt
system = """
You are a helpful assistant who faithfully answers user questions. Never respond with made up tool calls, only those you have been instructed to use.
"""

# https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
# https://langchain-ai.github.io/langgraph/#example
async def reactAgent():
    # create the agent    
    checkpointer = MemorySaver()
    agent_executor = create_react_agent(model, tools, state_modifier=system, checkpointer=checkpointer)

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="whats the weather in Minneapolis?")]}, config
    ):
        print(chunk)
        print("----")  
        
# Define the function that calls the model
def call_model(state: MessagesState):
    print("call_model")
    messages = state['messages']
    print(messages)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}     


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    print("should_continue")
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls and last_message.tool_calls[0]["name"] != "__main__":
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END
        
# https://langchain-ai.github.io/langgraph/#example
async def agentGraph():    
    # Define a new graph
    workflow = StateGraph(MessagesState)
    
    # tool node
    tool_node = ToolNode(tools)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.add_edge(START, "agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)

    # Use the Runnable
    config = {"configurable": {"thread_id": "abc123"}}
    for chunk in app.stream(
        {"messages": [
            SystemMessage(content=system),
            HumanMessage(content="whats the weather in Minneapolis?")
        ]}, config):
        print(chunk)
        print("----")
    
"""
TODO well, it seems like this model doesn't respond well to ToolMessages. It
always responds with additional made up tool calls.
"""
async def main():
    print("Hello from langchain-learning!")
    await agentGraph()
    # await reactAgent()

if __name__ == "__main__":
    asyncio.run(main())
