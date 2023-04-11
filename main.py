import os
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, download_loader
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import initialize_agent, Tool, ZeroShotAgent, AgentExecutor
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import ConversationBufferMemory, ChatMessageHistory, ReadOnlySharedMemory
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper

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

    tools = [
        Tool(
            name="Local Directory Index",
            func=lambda q: index.query(q),
            description=f"Useful when you want answer questions about the files in your local directory.",
        ),
        file_viewer,
    ]
    
    llm = OpenAI(temperature=config.temperature)
    
    memory = ConversationBufferMemory()
    memory.load_memory_variables({})
    
    agent_chain = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True)
    
    output = agent_chain.run(input="How can I implement an agi like system this project folder? First research agi systems like AutoGPT and BabyAGI on github. Then create a workspace folder within the project folder. Finally make your modifications and also log all changes with reasoning in the workspace folder.")
    history = ChatMessageHistory()
    
    print(output)
    print(history)

if __name__ == "__main__":
    main()