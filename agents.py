from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from typing import Dict, List, Optional, Any
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.agents import initialize_agent

#class ManagerAgent()

class CustomMRKLAgent(Chain):
    def __init__(self, llm: BaseLLM, vectorstore: FAISS, embeddings_model: OpenAIEmbeddings, tools: Dict[str, Any]):
        self.llm = llm
        self.vectorstore = vectorstore
        self.embeddings_model = embeddings_model
        self.task_creation_chain = TaskCreationChain.from_llm(llm)
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm(llm)
        self.execution_chain = ExecutionChain.from_llm(llm, tools=tools)

    def generate_tasks(self, result: str, task_description: str, incomplete_tasks: List[str], objective: str) -> List[str]:
        prompt = PromptTemplate(
            template="You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array.",
            input_variables=["result", "task_description", "incomplete_tasks", "objective"],
        )
        response = self.task_creation_chain(prompt.format(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective))
        return response.strip().split("\n")

    def prioritize_tasks(self, tasks: List[str], results: List[str]) -> List[str]:
        prompt = PromptTemplate(
            template="You are a task prioritization AI that takes in a list of tasks and their results"
            " and returns a prioritized list of tasks to be executed next."
            " These are the tasks: {tasks}."
            " These are the results: {results}."
            " Return the tasks as an array.",
            input_variables=["tasks", "results"],
        )
        response = self.task_prioritization_chain(prompt.format(tasks=tasks, results=results))
        return response.strip().split("\n")

    def execute_task(self, task: str) -> str:
        prompt = PromptTemplate(
            template="You are an execution AI that takes in a task and returns the result."
            " This is the task: {task}."
            " Return the result as a string.",
            input_variables=["task"],
        )
        response = self.execution_chain(prompt.format(task=task))
        return response.strip()