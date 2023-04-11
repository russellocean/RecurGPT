import os
# Importing the necessary modules
from langchain.tools import BaseTool
from config import config

# Defining a new tool class that inherits from BaseTool
class FileViewer(BaseTool):
    # Setting the name and description of the tool
    name = "File Viewer"
    description = "A tool that views a file and returns its contents"

    # Defining the main logic of the tool
    def _run(self, file_path: str) -> str:
        # Constructing the full file path
        full_file_path = os.path.join(config.project_folder, file_path)

        # Opening the file in read mode
        with open(full_file_path, "r") as f:
            # Reading the file contents
            file_contents = f.read()

        # Checking if the file contents are over 4000 characters
        if len(file_contents) > 4000:
            # TODO: Use a different function to summarize the file contents
            pass

        # Returning the file contents as the output
        return file_contents

    # Defining the async version of the tool, which is not implemented in this case
    async def _arun(self, file_path: str) -> str:
        raise NotImplementedError("FileViewer does not support async")