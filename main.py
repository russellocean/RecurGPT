import os
from typing import Optional

from dotenv import load_dotenv
import chardet
import fnmatch

from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

from agent import BabyAGI
from agent_utils import ask_agent

from config import config
from utils import ask_user_for_project

# Load environment variables from .env file
load_dotenv()

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


# Load environment variables
load_dotenv()

def main():
    start_sequence()
    while True:
        question = input("What would you like the agent to do? (type 'quit' or 'exit' to close)\n")
        if question.lower() == 'quit' or question.lower() == 'exit':
            break
        
        OBJECTIVE = question
        
        llm = OpenAI(temperature=0)
        
        # Logging of LLMChains
        verbose=False
        # If None, will keep on going forever
        max_iterations: Optional[int] = 3
        baby_agi = BabyAGI.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            verbose=verbose,
            max_iterations=max_iterations
        )
        
        # Set up the agent_executor
        baby_agi.agent_executor = BabyAGI.setup_agent(llm=llm)
        
        baby_agi({"objective": OBJECTIVE})
        
        
def start_sequence():
    print("Welcome to RecurGPT please select your project folder.")
    
    config.project_folder = ask_user_for_project()
    
    # Read files from the directory
    project_folder = config.project_folder
    documents = read_files_from_directory(project_folder)
    
    # Split documents using TextSplitter
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 50,
    )
    document_objs = [Document(page_content=text) for text in documents]
    texts = text_splitter.split_documents(document_objs)
    
    print("Current project folder:" + config.project_folder)
    
# def chroma_search(query):
#     results = chroma_store.similarity_search(query)
#     # Process the results and return the desired output
#     return results

def read_gitignore(directory):
    gitignore_path = os.path.join(directory, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as file:
            return file.read().splitlines()
    return []

def is_ignored(file, gitignore):
    for pattern in gitignore:
        if fnmatch.fnmatch(file, pattern):
            return True
    return False

def read_files_from_directory(directory):
    documents = []
    gitignore = read_gitignore(directory)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        print(f"Reading file: {file_path}")
        if os.path.isfile(file_path) and not is_ignored(filename, gitignore):
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']
                print(f"Detected encoding: {encoding}")
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                documents.append(content)
                print(f"Read {len(content)} characters from file: {file_path}")
    return documents

if __name__ == "__main__":
    main()