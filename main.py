from agent_utils import ask_agent

def main():
    while True:
        question = input("What would you like the agent to do? (type 'quit' or 'exit' to close)\n")
        if question.lower() == 'quit' or question.lower() == 'exit':
            break
        
        OBJECTIVE = question
        ask_agent(OBJECTIVE)

if __name__ == "__main__":
    main()