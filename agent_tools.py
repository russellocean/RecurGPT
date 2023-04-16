import os

# Import necessary libraries and modules
from pydantic import BaseModel
from langchain.tools import BaseTool

class ListDirectoriesTool(BaseTool):
    name = "ListDirectories"
    description = "Lists directories in a specified location. Provide the full directory path as expected by os.path, not a relative path. Example usage: `ListDirectories('/path/to/directory')`"

    def _run(self, path: str) -> str:
        """Helper function to list directories."""

        # Check if the path is valid
        if not os.path.exists(path):
            return f"Error: The specified path '{path}' does not exist. Please provide a valid directory."

        # Check if the path is a directory
        if not os.path.isdir(path):
            print("Error: The specified path is not a directory. Please provide a valid directory.")
            return f"Error: The specified path '{path}' is not a directory. Please provide a valid directory."

        # Call APIs or perform main functionality
        try:
            directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            output = ", ".join(directories)
        except Exception as e:
            return f"Error: An unexpected error occurred while listing directories: {str(e)}"

        return output

    async def _arun(self) -> str:
        raise NotImplementedError("ListDirectoriesTool does not support async")
    
    
class ListFilesAndDirectoriesTool(BaseTool):
    name = "ListFilesAndDirectories"
    description = "Lists files and directories in a specified location. Provide the full directory path as expected by os.path, not a relative path."

    def _run(self, path: str) -> str:
        """Helper function to list files and directories."""

        # Check if the path is valid
        if not os.path.exists(path):
            return f"Error: The specified path '{path}' does not exist. Please provide a valid directory."

        # Check if the path is a directory
        if not os.path.isdir(path):
            return f"Error: The specified path '{path}' is not a directory. Please provide a valid directory."

        # Call APIs or perform main functionality
        try:
            items = os.listdir(path)
            directories = [d for d in items if os.path.isdir(os.path.join(path, d))]
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]
            output_directories = "Directories: " + ", ".join(directories)
            output_files = "Files: " + ", ".join(files)
            output = output_directories + "\n" + output_files
        except FileNotFoundError as e:
            return f"Error: The specified directory '{path}' does not exist."
        except PermissionError as e:
            return f"Error: You do not have permission to access the directory '{path}'."
        except TypeError as e:
            return f"Error: An internal function was called with an invalid argument."
        except Exception as e:
            return f"Error: An unexpected error occurred while listing files and directories: {str(e)}"

        return output
    
    async def _arun(self) -> str:
        raise NotImplementedError("ListFilesAndDirectoriesTool does not support async")

class ViewCodeFilesTool(BaseTool):
    name = "ViewCodeFiles"
    description = "Views code files in a specified location. Provide the full file path as expected by os.path, not a relative path. Example usage: `ViewCodeFiles('/path/to/file.txt')`"

    def _run(self, file_path: str) -> str:
        """Helper function to view code files."""

        # Check if the path is valid
        if not os.path.exists(file_path):
            return f"Error: The specified path '{file_path}' does not exist. Please provide a valid file path."

        # Check if the path is a file
        if not os.path.isfile(file_path):
            return f"Error: The specified path '{file_path}' is not a file. Please provide a valid file path."

        # Call APIs or perform main functionality
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            output = f"Content of '{file_path}':\n{content}"
        except FileNotFoundError as e:
            return f"Error: The specified file '{file_path}' does not exist."
        except PermissionError as e:
            return f"Error: You do not have permission to access the file '{file_path}'."
        except UnicodeDecodeError as e:
            return f"Error: The file '{file_path}' contains non-text content or is not a supported coding file format."
        except Exception as e:
            return f"Error: An unexpected error occurred while reading the file: {str(e)}"

        return output

    async def _arun(self) -> str:
        raise NotImplementedError("ViewCodeFilesTool does not support async")
    
class CreateFileTool(BaseTool):
    name = "CreateFile"
    description = "Creates a new file at the specified location. Example usage: `CreateFile('/path/to/file.txt')`"

    def _run(self, file_path: str) -> str:
        """Helper function to create a new file."""

        # Check if the path already exists
        if os.path.exists(file_path):
            return f"Error: The specified path '{file_path}' already exists. Please provide a new file path."

        # Create the file
        try:
            with open(file_path, 'w') as file:
                file.write('')
            output = f"File '{file_path}' has been created successfully."
        except FileNotFoundError as e:
            return f"Error: The specified directory for the file '{file_path}' does not exist."
        except PermissionError as e:
            return f"Error: You do not have permission to create the file '{file_path}'."
        except Exception as e:
            return f"Error: An unexpected error occurred while creating the file: {str(e)}"

        return output

    async def _arun(self) -> str:
        raise NotImplementedError("CreateFileTool does not support async")

class ModifyFileTool(BaseTool):
    name = "ModifyFile"
    description = "Modifies the content of a file at the specified location. The input should be a string with the file path and new content separated by a comma. For example, 'path/to/file.txt, new content'."

    def _run(self, inputs: str) -> str:
        """Helper function to modify a file."""
        file_path, content = self.parse_inputs(inputs)

        # Check if the path is valid
        if not os.path.exists(file_path):
            return f"Error: The specified path '{file_path}' does not exist. Please provide a valid file path."

        # Check if the path is a file
        if not os.path.isfile(file_path):
            return f"Error: The specified path '{file_path}' is not a file. Please provide a valid file path."

        # Call APIs or perform main functionality
        try:
            with open(file_path, 'w') as file:
                file.write(content)
            output = f"File '{file_path}' has been modified successfully."
        except FileNotFoundError as e:
            return f"Error: The specified file '{file_path}' does not exist."
        except PermissionError as e:
            return f"Error: You do not have permission to modify the file '{file_path}'."
        except Exception as e:
            return f"Error: An unexpected error occurred while modifying the file: {str(e)}"

        return output

    def parse_inputs(self, inputs: str) -> tuple:
        file_path, content = inputs.split(",", 1)
        return file_path.strip(), content.strip()

    async def _arun(self) -> str:
        raise NotImplementedError("ModifyFileToolWithParser does not support async")

  
# class ModifyFileTool(BaseTool):
#     name = "ModifyFile"
#     description = "Modifies the content of a file at the specified location. Example usage: ('file_path': '<path/to/file>' , 'content': '<new content>')"

#     def _run(self, file_path: str, content: str) -> str:
#         """Helper function to modify a file."""

#         # Check if the path is valid
#         if not os.path.exists(file_path):
#             return f"Error: The specified path '{file_path}' does not exist. Please provide a valid file path."

#         # Check if the path is a file
#         if not os.path.isfile(file_path):
#             return f"Error: The specified path '{file_path}' is not a file. Please provide a valid file path."

#         # Call APIs or perform main functionality
#         try:
#             with open(file_path, 'w') as file:
#                 file.write(content)
#             output = f"File '{file_path}' has been modified successfully."
#         except FileNotFoundError as e:
#             return f"Error: The specified file '{file_path}' does not exist."
#         except PermissionError as e:
#             return f"Error: You do not have permission to modify the file '{file_path}'."
#         except Exception as e:
#             return f"Error: An unexpected error occurred while modifying the file: {str(e)}"

#         return output

#     async def _arun(self) -> str:
#         raise NotImplementedError("ModifyFileTool does not support async")

# Define the tool class
# class MRKLDirectAPIWrapper(BaseModel):
#     def _list_directories(self) -> str:
#         """Helper function to list directories."""
#         tool = ListDirectoriesTool()
#         return tool.run(self.path)

#     def _view_code_files(self) -> str:
#         """Helper function to view code files."""
#         tool = ViewCodeFilesTool()
#         return tool.run(self.file_path)

#     def _modify_code_files(self) -> str:
#         """Helper function to modify code files."""
#         tool = ModifyFileTool()
#         return tool.run(self.file_path, self.new_content)

#     def _self_correct_output(self) -> str:
#         """Helper function to self-correct output."""
#         # Implement the self-correction logic here
#         return "This feature is not implemented yet."

#     def _search_internet(self) -> str:
#         """Helper function to search the internet for documentation and code repositories."""
#         # Implement the internet search logic here
#         return "This feature is not implemented yet."

#     def run(self, query: str) -> str:
#         # Main method for running the tool
#         if "list directories" in query:
#             return self._list_directories()
#         elif "view code files" in query:
#             return self._view_code_files()
#         elif "modify code files" in query:
#             return self._modify_code_files()
#         elif "self correct output" in query:
#             return self._self_correct_output()
#         elif "search documentation and code repositories" in query:
#             return self._search_internet()
#         else:
#             raise ValueError("Invalid query")
        ...

# class MRKLDirectRun(BaseTool):
#     """The MRKLDirect tool uses the MRKL logic for listing directories, viewing code files, modifying them based on user requests, self-correcting its output, and searching the internet for documentation and code repositories."""

#     api_wrapper: MRKLDirectAPIWrapper
#     name = "MRKLDirect"
#     description = (
#         """The MRKLDirect tool can be used to perform the following tasks:
#         - Listing directories in a specified location.
#         - Viewing code files in a specified location.
#         - Modifying the content of a code file in a specified location.
#         Provide a query string as input to perform the desired task. The tool supports full directory paths as expected by os.path, including paths with the `~` character to represent the user's home directory.
#         """
#     )
#     verbose = True
#     #final_answer_format = "Final answer: the output message of the tool based on the performed task"

#     def _run(self, query: str) -> str:
#         """Use the tool."""
#         api_wrapper = MRKLDirectAPIWrapper.parse_raw(query)
#         api_wrapper.path = os.path.expanduser(api_wrapper.path)
#         api_wrapper.file_path = os.path.expanduser(api_wrapper.file_path)
#         return api_wrapper.run()

#     async def _arun(self, query: str) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError("MRKLDirectRun does not support async")