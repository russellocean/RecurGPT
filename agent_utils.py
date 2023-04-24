import os
from datetime import datetime
from typing import List, Optional

import faiss
import openai
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import AutoGPT
from langchain.experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
)
from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.schema import AIMessage, Document, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import BaseModel, Field, ValidationError

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

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


def setup_agent(context, project_directory):
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
            description="Useful for answering questions about current events. Ask targeted questions.",
        ),
        Tool(
            name="Wolfram",
            func=wolfram.run,
            description="Useful for answering questions about math, science, and geography.",
        ),
        Tool(
            name="Context",
            func=context.as_retriever,
            description="Useful for answering questions about the current project, within the context of the files. Ask targeted questions.",
        ),
    ]

    tools = [
        list_files_and_directories_tool,
        view_code_files_tool,
        create_file_tool,
        modify_file_tool,
    ] + tools

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prefix = f"""The current date and time is {current_datetime}. 
    You must utilize the tools given to you to best complete the task.
    Whenever completing a task, always follow this structure:

    1. Understand the context: Extract key information from the input to grasp the context, requirements, and goals of the task.
    2. Analyze the codebase: If provided, examine the existing codebase or project files to understand the structure, design patterns, and coding conventions used.
    3. Research and plan: Conduct research, if needed, to familiarize yourself with the relevant concepts, frameworks, or tools mentioned in the task.
    4. Break down the problem: Identify subtasks or components that need to be addressed to fulfill the requirements.
    5. Provide solutions: Based on the analysis and understanding of the problem, provide targeted solutions, code snippets, or explanations that address the specific requirements.
    6. Offer guidance and best practices: Suggest improvements or best practices to ensure the proposed solution is maintainable, efficient, and consistent with the project's style and format.
    7. Address potential challenges: Mention any potential issues or challenges that may arise during the implementation and suggest ways to overcome them.

    Here are the tools available to you; use them to complete the objective above and follow the 7 methods described:
    """

    suffix = f"""\nCurrent Project Directory: {project_directory}"""

    # llm = ChatOpenAI(temperature=0, model="gpt-4")
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    agent = CustomAutoGPT.from_llm_and_tools_custom(
        prefix=prefix,
        suffix=suffix,
        ai_name="DeveloperAgent",
        ai_role="A software development assistant",
        memory=vectorstore.as_retriever(),
        tools=tools,
        llm=llm,
    )

    # Set verbose to be true
    agent.chain.verbose = True

    return agent


class ToolAction(BaseModel):
    question: str = Field(description="The input question to answer")
    thought: str = Field(description="The thought about what to do")
    action: str = Field(description="The action to take")
    action_input: Optional[str] = Field(description="The input to the action")
    observation: Optional[str] = Field(description="The result of the action")
    criticism: Optional[str] = Field(
        description="Potential issues or limitations of the solution"
    )
    final_answer: Optional[str] = Field(
        description="The final answer to the original input question"
    )


class ToolActionsOutput(BaseModel):
    actions: List[ToolAction] = Field(description="List of tool actions")


def ask_agent(agent, message):
    """
    Run the agent with the provided message and return its response.
    """
    response = agent.run([message])

    return response


class CustomAutoGPT(AutoGPT):
    def run(self, goals: List[str]) -> str:
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )
        loop_count = 0
        while True:
            loop_count += 1

            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.full_message_history,
                memory=self.memory,
                user_input=user_input,
            )

            print(assistant_reply)
            self.full_message_history.append(HumanMessage(content=user_input))
            self.full_message_history.append(AIMessage(content=assistant_reply))

            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                return action.args["response"]
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = f"Error in args: {str(e)}"
                result = f"Command {tool.name} returned: {observation}"
                criticism = action.args.get("criticism")
                if criticism:
                    result += f"\nCriticism: {criticism}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                memory_to_add += feedback

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))

    @classmethod
    def from_llm_and_tools_custom(
        cls,
        prefix: str,
        suffix: str,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[AutoGPTOutputParser] = None,
    ) -> "CustomAutoGPT":
        custom_prompt = CustomAutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
            prefix=prefix,
            suffix=suffix,
        )
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=custom_prompt)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            feedback_tool=human_feedback_tool,
        )


class CustomAutoGPTPrompt(AutoGPTPrompt):
    prefix: str
    suffix: str

    def construct_full_prompt(self, goals: List[str]) -> str:
        # Call the parent class's construct_full_prompt method to get the original prompt
        original_prompt = super().construct_full_prompt(goals)

        # Add the prefix and suffix
        full_prompt = self.prefix + original_prompt + self.suffix

        return full_prompt
