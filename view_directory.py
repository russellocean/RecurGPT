import os
from typing import List
from langchain.agents import tool

@tool
def view_directory(path: str) -> List[str]:
    """
    View the contents of a directory given a path.
    """
    try:
        contents = os.listdir(path)
        return contents
    except FileNotFoundError:
        return f"Directory not found: {path}"
    except Exception as e:
        return f"Error: {e}"