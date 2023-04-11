import tkinter as tk
from tkinter import filedialog
from config import config

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory()  # Open folder selection dialog
    return folder_path

def ask_user_for_project():
    # If the PROJECT_FOLDER variable is already set, return it
    if config.project_folder is not None:
        print("Using project folder from config.py:", config.project_folder)
        return config.project_folder

    # Otherwise, ask the user to select a project folder
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory()  # Open folder selection dialog
    print("Selected folder:", folder_path)

    config.project_folder = folder_path
    return folder_path
