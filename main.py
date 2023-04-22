from agent_utils import ask_agent, setup_agent
from file_utils import (
    create_FAISS_vectorstore,
    load_documents_from_repository,
    select_ignore_file,
    select_project_repository,
)


def main():
    print("Welcome to the RecurGPT! Lets begin with selecting a project repository")

    project_repository = select_project_repository()
    print(f"Project repository selected: {project_repository}")
    ignore_file = select_ignore_file(initial_dir=project_repository)
    print(f"Ignore file selected: {ignore_file}")

    documents = load_documents_from_repository(project_repository, ignore_file)
    # preview_documents(documents) # Uncomment this line to preview the documentss
    vectorstore = create_FAISS_vectorstore(documents)

    # docsearch = chroma_vectorize(documents)

    # Create a vectorstore agent
    # project_knowledge = RetrievalQAWithSourcesChain.from_chain_type(
    #     OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
    # )

    # Setup the agent
    agent = setup_agent(vectorstore, project_repository)

    print("Project repository selected: " + project_repository)
    print("Now, lets begin with the agent!")

    while True:
        question = input(
            "What would you like the agent to do? (type 'quit' or 'exit' to close)\n"
        )
        if question.lower() == "quit" or question.lower() == "exit":
            break

        OBJECTIVE = question
        response = ask_agent(agent, OBJECTIVE)

        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
