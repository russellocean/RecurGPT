import json
import os

import openai
from langchain.agents import AgentType, Tool, ZeroShotAgent, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

# Import custom components
from agent_tools import (CreateFileTool, ListFilesAndDirectoriesTool,
                         ModifyFileTool, ViewCodeFilesTool)

# Retrieve API keys and app ID from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
WOLFRAM_ALPHA_APPID = os.environ.get("WOLFRAM_ALPHA_APPID")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def setup_agent():
    """
    Set up and return an instance of the agent.
    """
    
    # Initialize API wrappers
    search = GoogleSerperAPIWrapper()
    wolfram = WolframAlphaAPIWrapper()
    
    # Initialize custom tools
    list_files_and_directories_tool = ListFilesAndDirectoriesTool()
    view_code_files_tool = ViewCodeFilesTool()
    create_file_tool = CreateFileTool()
    modify_file_tool = ModifyFileTool()

    # Define available tools
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for answering questions about current events. Ask targeted questions."
        ),
        Tool(
            name="Wolfram",
            func=wolfram.run,
            description="Useful for answering questions about math, science, and geography."
        )
    ]
    
    tools = [list_files_and_directories_tool, view_code_files_tool, create_file_tool, modify_file_tool] + tools
    
    prefix = """You are an AI that is tasked with the objective of {objective}.\n\
    Your goal is to effectively coordinate the execution of tasks. Your responsibilities include:\n\
    1. Creating new tasks based on the results of previous tasks.\n\
    2. Prioritizing tasks based on the results of previous tasks.\n\
    3. Executing tasks.\n\
    Adhere to professional standards and best practices during the process.\n\
    Always monitor your overall progress towards the objective.
    
    Provide the final answer with Final Answer: <answer>"""
    
    suffix = """Begin!
    
    Objective: {objective}
    {agent_scratchpad}
    """
    
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "agent_scratchpad"]    
    )
    
    print(prompt.template)
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)

    return agent

def ask_agent(message):
    """
    Run the agent with the provided message and return its response.
    """
    
    agent = setup_agent()
    response = agent({"input": message})
    
    print(response["intermediate_steps"])
        
    print(json.dumps(response["intermediate_steps"], indent=2))
    return response