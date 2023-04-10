import os
import sys
import logging
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# Load environment variables
load_dotenv()

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    print("Welcome to RecurGPT please select your project folder.")
    # Ask user to select their repository
    project_folder = select_folder()
    print("Selected folder:", project_folder)
    
    # Load the project
    loader = SimpleDirectoryReader(project_folder, recursive=True, exclude_hidden=True).load_data()
    project = loader.load_data()
    index = GPTSimpleVectorIndex(project)
    
    # Query the project
    response = index.query("What does the main.py file do?")
    
    print(response)
    
    
def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory()  # Open folder selection dialog
    return folder_path

if __name__ == "__main__":
    main()