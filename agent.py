from typing import Dict, List, Optional, Any

from collections import deque
from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from langchain.utilities import GoogleSerperAPIWrapper, WolframAlphaAPIWrapper
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool

from agent_components import CustomOutputParser, CustomPromptTemplate

class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["result", "task_description", "incomplete_tasks", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
# Update the execution_template in ExecutionChain
class ExecutionChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "Answer the following question as best you can. You have access to the following tools:"
            " {tools}."
            " Use the following format:"
            " Thought: you should always think about what to do"
            " Action: the action to take, should be one of [{tool_names}]"
            " Action Input: the input to the action"
            " Observation: the result of the action"
            " ... (this Thought/Action/Action Input/Observation can repeat N times)"
            " Thought: I now know the final answer"
            " Final Answer: the final answer to the original input question"
            " If answering a coding question only provide code, add comments to communicate your thought process."
            " Begin!"
            " Question: {task}"
            " Context: {context}"
            " Objective: {objective}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["tools", "tool_names", "objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
# Add a new method to execute tasks using tools
def execute_task_with_tools(execution_chain: LLMChain, tools: List[Tool], objective: str, task: str, context: List[str]) -> str:
    """Execute a task using tools."""
    tool_names = [tool.name for tool in tools]
    tools_description = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    return execution_chain.run(tools=tools_description, tool_names=", ".join(tool_names), objective=objective, context=context, task=task)

# Update the execute_task function to call the new execute_task_with_tools method
def execute_task(vectorstore, execution_chain: LLMChain, tools: List[Tool], objective: str, task: str, k: int = 5) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execute_task_with_tools(execution_chain, tools, objective, task, context)
    
def get_next_task(task_creation_chain: LLMChain, result: Dict, task_description: str, task_list: List[str], objective: str) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

def prioritize_tasks(task_prioritization_chain: LLMChain, this_task_id: int, task_list: List[Dict], objective: str) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(task_names=task_names, next_task_id=next_task_id, objective=objective)
    new_tasks = response.split('\n')
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list

def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata['task']) for item in sorted_results]

class BabyAGI(Chain, BaseModel):
    agent_executor: Optional[AgentExecutor] = Field(None)
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None
        
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)
        
    @property
    def input_keys(self) -> List[str]:
        return ["objective"]
    
    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs['objective']
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = self.agent_executor.run(task["task_name"]) if self.agent_executor else execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain, this_task_id, list(self.task_list), objective
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break
        return {}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        execution_chain = ExecutionChain.from_llm(llm, verbose=verbose)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs
        )
        
    @classmethod
    def setup_agent(cls, llm: BaseLLM):
        # Initialize API wrappers
        search = GoogleSerperAPIWrapper()
        wolfram = WolframAlphaAPIWrapper()

        # Define available tools
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Useful for when you need to answer questions about current events. You should ask targeted questions"
            ),
            Tool(
                name="Wolfram",
                func=wolfram.run,
                description="Useful for when you need to answer questions about math, science, geography."
            )
        ]

        # Set up the base template for the agent
        template = """Answer the following questions as best you can. You have access to the following tools:
        {tools}
        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        If answering a coding question only provide code, add comments to communicate your thought process.

        Begin!
        Question: {input}
        {agent_scratchpad}"""

        # Set up the custom prompt template
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            input_variables=["input", "intermediate_steps"]
        )

        # Initialize custom output parser
        output_parser = CustomOutputParser()

        # Set up the LLMChain
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Extract tool names for the agent
        tool_names = [tool.name for tool in tools]

        # Set up the LLMSingleActionAgent
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        # Set up the AgentExecutor
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

        return agent_executor
