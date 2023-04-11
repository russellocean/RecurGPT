def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory()  # Open folder selection dialog
    return folder_path

def ask_user_for_project():
    # Ask user to select their repository
    project_folder = select_folder()
    print("Selected folder:", project_folder)