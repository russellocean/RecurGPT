import os
from langchain.llms import OpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Config(metaclass=Singleton):
    def __init__(self):
        self.project_folder = None
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.temperature = 0.9

# Create a Config instance
config = Config()