import json
import os
from typing import List, Optional

import openai
from langchain import LLMChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.utilities import GoogleSerperAPIWrapper, TextRequestsWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from pydantic import BaseModel, Field

# Import custom components
from agent_tools import (
    CreateFileTool,
    ListFilesAndDirectoriesTool,
    ModifyFileTool,
    ViewCodeFilesTool,
)

# Retrieve API keys and app ID from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
WOLFRAM_ALPHA_APPID = os.environ.get("WOLFRAM_ALPHA_APPID")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY


def setup_agent(context, project_directory):
    """
    Set up and return an instance of the agent.
    """

    # Initialize API wrappers
    search = GoogleSerperAPIWrapper()
    wolfram = WolframAlphaAPIWrapper()
    TextRequestsWrapper()

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
            description="Useful for answering questions about current events. Ask targeted questions.",
        ),
        Tool(
            name="Wolfram",
            func=wolfram.run,
            description="Useful for answering questions about math, science, and geography.",
        ),
        # Tool(
        #     name="Requests",
        #     func=requests.run,
        #     description="Useful for fetch data from a website.",
        # ),
        Tool(
            name="Context",
            func=context.run,
            description="Useful for when you need information about the current project. Can be used to answer questions about the project's codebase, documentation, and more. Use in the form of a question.",
        ),
    ]

    tools = [
        list_files_and_directories_tool,
        view_code_files_tool,
        create_file_tool,
        modify_file_tool,
    ] + tools

    memory = ConversationBufferMemory()

    prefix = """You are an AI that is tasked with the objective of {input}.\n\
    Your goal is to effectively coordinate the execution of tasks. Your responsibilities include:\n\
    1. Creating new tasks based on the results of previous tasks.\n\
    2. Prioritizing tasks based on the results of previous tasks.\n\
    3. Executing tasks.\n\
    Adhere to professional standards and best practices during the process.\n\
    Always monitor your overall progress towards the objective.

    If you wish to record your thoughts use a file.\n\
    unless if you are providing the final answer to the objective.\n\
    
    Provide your current completion status with Progress: <progress>
    Provide the final answer with Final Answer: <answer>
    """

    # suffix = """Begin!

    # {chat_history}
    # Current Project Directory: {project_directory}
    # Objective: {input}
    # {agent_scratchpad}
    # """
    suffix = (
        "Begin!\n\nCurrent Project Directory: "
        + project_directory
        + "\nObjective: {input}\n{agent_scratchpad}"
        + "Alawys follow the format, never send a message without a format."
    )

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
    )

    print(prompt.template)

    llm = ChatOpenAI(temperature=0, model="gpt-4")

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    parser = PydanticOutputParser(pydantic_object=ToolActionsOutput)
    fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory, output_parser=fix_parser
    )

    # agent = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     return_intermediate_steps=True,
    # )

    return agent_chain


class ToolAction(BaseModel):
    question: str = Field(description="The input question to answer")
    thought: str = Field(description="The thought about what to do")
    action: str = Field(description="The action to take")
    action_input: Optional[str] = Field(description="The input to the action")
    observation: Optional[str] = Field(description="The result of the action")
    final_answer: Optional[str] = Field(
        description="The final answer to the original input question"
    )


class ToolActionsOutput(BaseModel):
    actions: List[ToolAction] = Field(description="List of tool actions")


def ask_agent(agent, message):
    """
    Run the agent with the provided message and return its response.
    """

    # agent = setup_agent()
    response = agent.run(message)
    # response = agent({"input": message})

    print(response["intermediate_steps"])

    print(json.dumps(response["intermediate_steps"], indent=2))
    return response
