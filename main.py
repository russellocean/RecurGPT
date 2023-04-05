import os
import openai
import pinecone
import sys
import logging
from collections import deque
from dotenv import load_dotenv
from colorama import Fore, Back, Style, init

# Load environment variables
load_dotenv()

# Initialize colorama
init(autoreset=True)

# Initialize OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
YOUR_TABLE_NAME = "test"

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east4-gcp")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_pinecone_index(table_name, dimension=1536, metric="cosine", pod_type="p1"):
    if table_name not in pinecone.list_indexes():
        pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        print(f"Pinecone index '{table_name}' created with dimension {dimension}, metric {metric}, and pod_type {pod_type}.")
    else:
        print(f"Pinecone index '{table_name}' already exists.")


# Global variables
agents = {}
next_key = 1

def create_agent(task, prompt, model="gpt-3.5-turbo"):
    global next_key
    global agents

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    agent_reply = openai_call(prompt, model=model, messages=messages)
    messages.append({"role": "assistant", "content": agent_reply})

    key = next_key
    next_key += 1

    agents[key] = (task, messages, model)
    return key, agent_reply

def message_agent(key, message):
    global agents

    task, messages, model = agents[int(key)]
    messages.append({"role": "user", "content": message})

    agent_reply = openai_call(message, model=model, messages=messages)
    messages.append({"role": "assistant", "content": agent_reply})

    return agent_reply

def list_agents():
    global agents
    return [(key, task) for key, (task, _, _) in agents.items()]

def delete_agent(key):
    global agents

    try:
        del agents[int(key)]
        return True
    except KeyError:
        return False

def openai_call(prompt, model="gpt-3.5-turbo", messages=None, temperature=0.5, max_tokens=100):
    logging.info(f"Sending prompt to OpenAI: {prompt}")
    confirm = input("Press 'y' to confirm OpenAI call or any other key to exit: ")

    if confirm.lower() != 'y':
        sys.exit("User exited the program.")

    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    result = response['choices'][0]['message']['content'].strip()
    logging.info(f"OpenAI response: {result}")
    return result


def task_creation_agent(objective, result=None, task_description=None, task_list=None):
    if not task_list:
        task_list = []
    if result and task_description:
        prompt = f"You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}. The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {''.join(task_list)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array."

    else:
        prompt = f"Create a list of tasks to accomplish the following objective: {objective}. Return the tasks as an array."
    response = openai_call(prompt)
    return response


def execution_agent(objective, task):
    context = context_agent(index=YOUR_TABLE_NAME, query=objective, n=5)
    prompt = f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"
    return openai_call(prompt, temperature=0.7, max_tokens=2000)


def context_agent(query, index, n):
    query_embedding = get_ada_embedding(query)
    index = pinecone.Index(index_name=index)
    results = index.query(query_embedding, top_k=n, include_metadata=True)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata['task'])) for item in sorted_results]


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


def fetch_result(task):
    task_str = str(task)
    task_id = str(hash(task_str))
    result = index.fetch(ids=[task_id])
    if task_id in result:
        return result[task_id]['metadata']['result']
    return None


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

def main(objective):
    global task_list
    task_list_response = task_creation_agent(objective)
    print(f"{Fore.GREEN}Task list response: {task_list_response}{Style.RESET_ALL}")

    while True:
        task_prompt = f"Given the following list of tasks, pick one task to perform:\n{task_list_response}\nSelected task:"
        task_name = openai_call(task_prompt, max_tokens=200).strip()
        if not task_name:
            break

        print(f"{Fore.YELLOW}Selected task: {task_name}{Style.RESET_ALL}")

        # Create a new agent and delegate the task
        agent_key, agent_reply = create_agent(task_name, task_name)
        print(f"{Fore.MAGENTA}Agent created with key: {agent_key}, initial reply: {agent_reply}{Style.RESET_ALL}")

        # List the agents
        agent_list = list_agents()
        print(f"{Fore.MAGENTA}Agent list: {agent_list}{Style.RESET_ALL}")

        # Message the agent to perform the task
        result = message_agent(agent_key, task_name)
        print(f"{Fore.CYAN}Result of the execution agent: {result}{Style.RESET_ALL}")

        save_result(task_name, result)

        # Update the task list based on the result
        new_tasks_response = task_creation_agent(objective, result, task_name, task_list_response)
        if new_tasks_response != task_list_response:
            task_list_response = new_tasks_response
            print(f"{Fore.GREEN}Updated task list response: {task_list_response}{Style.RESET_ALL}")

        # Delete the agent
        agent_deleted = delete_agent(agent_key)
        if agent_deleted:
            print(f"{Fore.MAGENTA}Agent with key {agent_key} deleted.{Style.RESET_ALL}")


if __name__ == "__main__":
    create_pinecone_index(YOUR_TABLE_NAME)
    index = pinecone.Index(index_name=YOUR_TABLE_NAME)
    objective = "Design a basic website layout"
    main(objective)
    pinecone.deinit()