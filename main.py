from dotenv import load_dotenv

from llama_index import download_loader
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

from project_loader import load_project
from utils import ask_user_for_project

from config import config

# Importing the FileViewer tool from file_viewer.py
from file_viewer import FileViewer

# Load environment variables
load_dotenv()

SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

def main():
    print("Welcome to RecurGPT please select your project folder.")
    config.project_folder = ask_user_for_project()
    
    # Load the project
    index = load_project(config.project_folder)
    
    print("Current project folder:" + config.project_folder)
    
    # Creating an instance of the FileViewer tool
    file_viewer = FileViewer()

    #llm = OpenAI(model_name="text-davinci-003", streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0.5)
    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.8)
    
    
    
    directory_reader = Tool(
        name="Local Directory Index",
        func=lambda q: index.query(q),
        description=f"Useful when you want answer questions about the files in your local directory.",
    )
    
   # tools = load_tools(["Local Directory Index"],["human"], ["llm-math"], ["View File"], llm=llm)
    tools = load_tools(['human', 'wikipedia'], llm=llm)
    
    tools = tools + [file_viewer, directory_reader]
    
    memory = ConversationBufferMemory()
    memory.load_memory_variables({})
    
    agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True)
    
    agent.run("Describe this project folder.")
    
    #agent_chain = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True)
    
    #agent_chain.run(input="How can I implement an agi like system this project folder? First research agi systems like AutoGPT and BabyAGI on github. Then create a workspace folder within the project folder. Finally make your modifications and also log all changes with reasoning in the workspace folder.")

if __name__ == "__main__":
    main()