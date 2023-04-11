from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator

def load_project(project_folder):
    # Load the project
    loader = DirectoryLoader(project_folder, loader_cls=TextLoader)
    files = loader.load()
    print("Size of project:", len(files), "files.")

    index = VectorstoreIndexCreator().from_loaders([loader])
    return index
