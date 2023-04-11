from dotenv import load_dotenv
import os
import fnmatch
import chardet

from config import config
from utils import ask_user_for_project
from project_loader import load_project
from file_viewer import FileViewer
from llama_index import download_loader
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import OutputFixingParser
from custom_parser import CustomOutputFixingParser



from langchain.output_parsers import StructuredOutputParser, ResponseSchema


from view_directory import view_directory

import json


# Load environment variables
load_dotenv()

SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

def main():
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
    
    #Create Chroma vector database (assuming you have documents and embeddings)
    embeddings = OpenAIEmbeddings()
    chroma_store = Chroma.from_documents(texts, embeddings, collection_name="your-collection-name")
    
    # Load the project
    index = load_project(config.project_folder)
    
    print("Current project folder:" + config.project_folder)
    
    # Creating an instance of the FileViewer tool
    file_viewer = FileViewer()
    
    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.2, verbose=True)
    
    directory_reader = Tool(
        name="Local Directory Index",
        func=lambda q: index.query(q),
        description=f"Useful when you want answer questions about the files in your local directory.",
    )
    chroma_tool = Tool(
        name="Chroma Search",
        func=chroma_search,
        description="Useful for searching information based on similarity in the Chroma vector database."
    )
    

    # How you would like your reponse structured. This is basically a fancy prompt template
    response_schemas = [
        ResponseSchema(name="final_answer", description="The final answer to the question.")
    ]

    # How you would like to parse your output
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    
    fix_parser = OutputFixingParser.from_llm(parser=output_parser, llm=ChatOpenAI())
    
    custom_parser = CustomOutputFixingParser(llm=llm)

    
    template = """
    The users project folder is {project_folder}.
    The user is asking: {human_input}
    
    {format_instructions}
    
    Just provide the answer to the question, dont worry about the formatting.
    """

    prompt = PromptTemplate(
        input_variables=["project_folder", "human_input"],
        partial_variables={"format_instructions": format_instructions},
        template=template
    )
    
    promptValue = prompt.format(project_folder=config.project_folder, human_input="What files are in this project folder?")
    
   # Initialize the react-docstore agent
    docstore = DocstoreExplorer(Wikipedia())
    react_tools = [
        Tool(
            name="Search",
            func=docstore.search,
            description="useful for when you need to ask with search"
        ),
        Tool(
            name="Lookup",
            func=docstore.lookup,
            description="useful for when you need to ask with lookup"
        )
    ]
    tools = load_tools(['wikipedia', 'requests_all'], llm=llm)
    
    #tools = tools + [view_directory, file_viewer, chroma_tool] + react_tools
    
    tools = tools + [view_directory] + react_tools
    
    # memory = ConversationBufferMemory()
    # memory.load_memory_variables({})
    # Create shared memory
    memory = ConversationBufferMemory()
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True, output_parser=custom_parser)
    
    # agent.run("Study this project folder, then create a list of project names that would best fit. Be creative.")
    response = agent(promptValue)
    
    print(json.dumps(response["intermediate_steps"], indent=2))
    
    #agent_chain = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True)
    
    #agent_chain.run(input="How can I implement an agi like system this project folder? First research agi systems like AutoGPT and BabyAGI on github. Then create a workspace folder within the project folder. Finally make your modifications and also log all changes with reasoning in the workspace folder.")
    
def chroma_search(query):
    results = chroma_store.similarity_search(query)
    # Process the results and return the desired output
    return results

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