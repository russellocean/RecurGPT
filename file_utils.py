import fnmatch
import os
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional

import nltk
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

nltk.download("averaged_perceptron_tagger")


def select_project_repository():
    root = tk.Tk()
    root.withdraw()  # Hide the main window to only show the file dialog
    folder_path = filedialog.askdirectory(title="Select the project repository")
    return folder_path


def select_ignore_file(initial_dir: Optional[str] = None):
    root = tk.Tk()
    root.withdraw()  # Hide the main window to only show the file dialog
    file_path = filedialog.askopenfilename(
        title="Select the ignore file",
        filetypes=[("All files", "*.*")],  # Show all files, including hidden files
        initialdir=initial_dir
        or os.path.expanduser(
            "~"
        ),  # Start at the given directory or user's home directory
    )
    return file_path if file_path else None


def load_documents_from_repository(folder_path: str, ignore_file: Optional[str] = None):
    ignore_patterns = read_gitignore_and_exclude(folder_path, ignore_file)
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


def read_gitignore_and_exclude(
    folder_path: str, ignore_file: Optional[str] = None
) -> List[str]:
    gitignore_path = os.path.join(folder_path, ".gitignore")
    exclude_path = "./exclude.txt"
    ignore_patterns = []

    file_paths = [gitignore_path, exclude_path]
    if ignore_file:
        file_paths.append(ignore_file)

    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, "r", errors="replace") as ignore_file:
                patterns = [
                    line.strip()
                    for line in ignore_file
                    if line.strip() and not line.startswith("#")
                ]
                ignore_patterns.extend(patterns)

    return ignore_patterns


def is_ignored(file_path: str, folder_path: str, ignore_patterns: List[str]) -> bool:
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(file_path, f"*{pattern}*"):
            print(f"File {file_path} is ignored.")
            return True
    print(f"File {file_path} is not ignored.")
    return False


def file_load(file_path: str) -> str:
    with open(file_path, "r") as file:
        content = file.read()
    return content


def preview_documents(documents, lines_to_preview=5):
    for idx, doc in enumerate(documents):
        print(f"Document {idx + 1}: {doc.metadata['source']}")
        content = doc.page_content.split("\n")
        preview_lines = content[:lines_to_preview]
        print("\n".join(preview_lines))
        print("\n---\n")


def chroma_vectorize(documents):
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    content = splitter.split_documents(documents)
    # splitter = CharacterTextSplitter()
    vector_store = Chroma.from_documents(
        content, embeddings, collection_name="project-repo"
    )
    return vector_store


class CustomUnstructuredFileLoader(UnstructuredFileLoader):
    def __init__(
        self, file_path: str, folder_path: str, ignore_patterns: List[str], **kwargs
    ):
        self.folder_path = folder_path
        self.ignore_patterns = ignore_patterns
        super().__init__(file_path, **kwargs)

    def load(self) -> List:
        if is_ignored(self.file_path, self.folder_path, self.ignore_patterns):
            return []

        try:
            elements = self._get_elements()
        except ValueError as e:
            _, file_extension = os.path.splitext(self.file_path)
            print(f"Error while loading file: {self.file_path}. Error: {e}")
            print(f"Unsupported file type: {file_extension}. Adding to exclude.txt.")
            with open("exclude.txt", "a") as exclude_file:
                exclude_file.write(f"{file_extension}\n")
            return []

        return elements

    def _get_elements(self) -> List:
        from langchain.docstore.document import Document
        from langchain.document_loaders import PyPDFLoader
        from unstructured.partition.auto import partition

        # Define code file extensions that you want to support
        code_file_extensions = {
            ".cpp",
            ".c",
            ".cs",
            ".py",
            ".js",
            ".java",
            ".rb",
            ".XML",
            ".manifest",
            ".html",
            ".css",
            ".php",
            ".sql",
            ".go",
            ".swift",
            ".ts",
            ".kt",
            ".rs",
            ".hs",
            ".scala",
            ".clj",
            ".lua",
            ".m",
            ".r",
            ".sh",
            ".bat",
            ".vb",
            ".pl",
            ".fs",
            ".ml",
            ".mli",
            ".erl",
            ".hrl",
            ".ex",
            ".exs",
            ".eex",
            ".leex",
            ".yml",
            ".yaml",
            ".json",
            ".toml",
            ".ini",
            ".conf",
            ".cfg",
            ".prefs",
            ".properties",
            ".asciidoc",
            ".adoc",
            ".asc",
            ".md",
            ".markdown",
            ".rst",
            ".txt",
            ".tex",
            ".bib",
            ".bibliography",
            ".bib",
        }

        _, file_extension = os.path.splitext(self.file_path)

        if file_extension.lower() in code_file_extensions:
            # Load code files as plain text
            with open(self.file_path, "r", encoding="utf-8") as file:
                content = file.read()
            document = Document(
                page_content=content, metadata={"source": self.file_path}
            )
            return [document]
        if file_extension.lower() == ".pdf":
            # Load PDF files using PyPDFLoader
            loader = PyPDFLoader(self.file_path)
            pages = loader.load()
            return pages
        else:
            # Use partition for other file types
            return partition(filename=self.file_path, **self.unstructured_kwargs)


class SafeRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def split_documents(self, documents):
        texts = []
        for doc in documents:
            try:
                texts.append(doc.page_content)
            except AttributeError:
                print(
                    "Warning: 'FigureCaption' object has no attribute 'page_content'. Skipping this document."
                )
                continue

        return super().split_documents(texts)
