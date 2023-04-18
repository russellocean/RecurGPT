import fnmatch
import os
import tkinter as tk
from tkinter import filedialog
from typing import List

import nltk
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

nltk.download('averaged_perceptron_tagger')

def select_project_repository():
    root = tk.Tk()
    root.withdraw()  # Hide the main window to only show the file dialog
    folder_path = filedialog.askdirectory(title="Select the project repository")
    return folder_path

def load_documents_from_repository(folder_path: str):
    ignore_patterns = read_gitignore_and_exclude(folder_path)
    loader = DirectoryLoader(
        path=folder_path,
        glob="**/*",
        load_hidden=True,
        recursive=True,
        loader_cls=CustomUnstructuredFileLoader,
        loader_kwargs={"folder_path": folder_path, "ignore_patterns": ignore_patterns},
    )
    documents = loader.load()
    return documents

def read_gitignore_and_exclude(folder_path: str) -> List[str]:
    gitignore_path = os.path.join(folder_path, '.gitignore')
    exclude_path = './exclude.txt'
    ignore_patterns = []

    for file_path in [gitignore_path, exclude_path]:
        if os.path.exists(file_path):
            with open(file_path, 'r') as ignore_file:
                patterns = [line.strip() for line in ignore_file if line.strip() and not line.startswith('#')]
                ignore_patterns.extend(patterns)

    return ignore_patterns

def is_ignored(file_path: str, folder_path: str, ignore_patterns: List[str]) -> bool:
    relative_file_path = os.path.relpath(file_path, folder_path)
    for pattern in ignore_patterns:
        if fnmatch.fnmatchcase(relative_file_path, pattern):
            return True
    return False

def file_load(file_path: str) -> str:
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def preview_documents(documents, lines_to_preview=5):
    for idx, doc in enumerate(documents):
        print(f"Document {idx + 1}: {doc.metadata['source']}")
        content = doc.page_content.split('\n')
        preview_lines = content[:lines_to_preview]
        print("\n".join(preview_lines))
        print("\n---\n")
        
def chroma_vectorize(documents):
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    content = splitter.split_documents(documents)
    #splitter = CharacterTextSplitter()
    vector_store = Chroma.from_documents(content, embeddings)
    return vector_store

class CustomUnstructuredFileLoader(UnstructuredFileLoader):
    def __init__(self, file_path: str, folder_path: str, ignore_patterns: List[str], **kwargs):
        self.folder_path = folder_path
        self.ignore_patterns = ignore_patterns
        super().__init__(file_path, **kwargs)

    def load(self) -> List:
        if is_ignored(self.file_path, self.folder_path, self.ignore_patterns):
            return []
        return super().load()

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        # Define code file extensions that you want to support
        code_file_extensions = {".cpp", ".c", ".cs", ".py", ".js", ".java", ".rb"}

        _, file_extension = os.path.splitext(self.file_path)

        if file_extension.lower() in code_file_extensions:
            # Load code files as plain text
            with open(self.file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return [content]
        else:
            # Use partition for other file types
            return partition(filename=self.file_path, **self.unstructured_kwargs)