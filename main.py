import os
import openai
import pinecone
import time
import sys
import logging
from collections import deque
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

def create_pinecone_index(table_name, dimension=1536, metric="cosine", pod_type="p1"):
    if table_name not in pinecone.list_indexes():
        pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        print(f"Pinecone index '{table_name}' created with dimension {dimension}, metric {metric}, and pod_type {pod_type}.")
    else:
        print(f"Pinecone index '{table_name}' already exists.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
YOUR_TABLE_NAME = "test"
#YOUR_TABLE_NAME = os.getenv("TABLE_NAME")

print("OPENAI_API_KEY: ", OPENAI_API_KEY)
print("PINECONE_API_KEY: ", PINECONE_API_KEY)
print("YOUR_TABLE_NAME: ", YOUR_TABLE_NAME)

if not OPENAI_API_KEY or not PINECONE_API_KEY or not YOUR_TABLE_NAME:
    print("One or more environment variables are missing. Please set OPENAI_API_KEY, PINECONE_API_KEY, and TABLE_NAME in your .env file.")
    sys.exit()

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east4-gcp")

# Create Pinecone index
create_pinecone_index(YOUR_TABLE_NAME)

index = pinecone.Index(index_name=YOUR_TABLE_NAME)

# Task list
task_list = deque([])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_task(task):
    task_list.append(task)

def openai_call(prompt, temperature=0.5, max_tokens=100):
    logging.info(f"Sending prompt to OpenAI: {prompt}")
    confirm = input("Press 'y' to confirm OpenAI call or any other key to exit: ")
    if confirm.lower() != 'y':
        sys.exit("User exited the program.")
        
    response = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None,
    )
    result = response.choices[0].text.strip()
    logging.info(f"OpenAI response: {result}")
    return result

def task_creation_agent(objective):
    prompt = f"Create a list of tasks to accomplish the following objective: {objective}. Return the tasks as an array."
    response = openai_call(prompt)
    tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in tasks]

def execution_agent(task):
    prompt = f"Perform the following task: {task}. Return the result."
    result = openai_call(prompt)
    return result

def save_result(task, result):
    task_str = str(task)
    task_id = str(hash(task_str))
    task_vector = get_ada_embedding(result)

    items = [
        {
            "id": task_id,
            "values": task_vector,
            "metadata": {"task": task_str, "result": result},
        }
    ]
    index.upsert(vectors=items)
    logging.info(f"Saved result for task '{task_str}': {result}")
    
def get_ada_embedding(text):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response["data"][0]["embedding"]


def main(objective):
    global task_list
    task_list = deque(task_creation_agent(objective))

    while task_list:
        task = task_list.popleft()
        result = execution_agent(task['task_name'])
        save_result(task, result)

    # pinecone.deinit()

if __name__ == "__main__":
    objective = "Design a basic website layout"
    main(objective)
