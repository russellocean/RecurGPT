from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

def load_project(project_folder):
    # Load the project
    
    print( "Loading project from folder:", project_folder)
    loader = DirectoryLoader(project_folder)
    files = loader.load()
    print("Size of project:", len(files), "files.")

    index = VectorstoreIndexCreator().from_loaders([loader])
    return index
