
from pyexpat import model
from dotenv import load_dotenv

from langchain import OpenAI

from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI




from agent_tools import ListFilesAndDirectoriesTool, ViewCodeFilesTool, CreateFileTool, ModifyFileTool

# Load environment variables from .env file
load_dotenv()

list_files_and_directories_tool = ListFilesAndDirectoriesTool()
view_code_files_tool = ViewCodeFilesTool()
create_file_tool = CreateFileTool()
modify_file_tool = ModifyFileTool()


def main():
    #start_sequence()
    tools = [list_files_and_directories_tool, view_code_files_tool, create_file_tool, modify_file_tool]
    
    prefix = """You are an AI that is tasked with the objective of {objective}.\n\
    Your goal is to effectively coordinate the execution of tasks. Your responsibilities include:\n\
    1. Creating new tasks based on the results of previous tasks.\n\
    2. Prioritizing tasks based on the results of previous tasks.\n\
    3. Executing tasks.\n\
    Adhere to professional standards and best practices during the process.\n\
    Always monitor your overall progress towards the objective."""


    
    suffix = """ 
        Begin! 
        
        Objective: {objective}
        {agent_scratchpad}
    """
    
    while True:
        question = input("What would you like the agent to do? (type 'quit' or 'exit' to close)\n")
        if question.lower() == 'quit' or question.lower() == 'exit':
            break
        
        OBJECTIVE = question
        
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix = prefix,
            suffix = suffix,
            input_variables=["objective", "agent_scratchpad"]    
        )
        
        print(prompt.template)
        
        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        tool_names = [tool.name for tool in tools]

        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, verbose=True, return_intermediate_steps=True)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        
        response = agent_executor.run(OBJECTIVE)
        
        print(response)

        # Logging of LLMChains
        verbose=False
        # If None, will keep on going forever
        
        
        
        
# def start_sequence():
#     print("Welcome to RecurGPT please select your project folder.")
    
#     config.project_folder = ask_user_for_project()
    
#     # Read files from the directory
#     project_folder = config.project_folder
#     documents = read_files_from_directory(project_folder)
    
#     # Split documents using TextSplitter
#     #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 1000,
#         chunk_overlap  = 50,
#     )
#     document_objs = [Document(page_content=text) for text in documents]
#     texts = text_splitter.split_documents(document_objs)
    
#     print("Current project folder:" + config.project_folder)
    

if __name__ == "__main__":
    main()