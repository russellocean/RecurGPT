from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

from agent_utils import ask_agent
from file_utils import (
    chroma_vectorize,
    load_documents_from_repository,
    preview_documents,
    select_ignore_file,
    select_project_repository,
)


def main():
    print("Welcome to the RecurGPT! Lets begin with selecting a project repository")

    project_repository = select_project_repository()
    ignore_file = select_ignore_file(initial_dir=project_repository)
    documents = load_documents_from_repository(project_repository, ignore_file)
    preview_documents(documents)

    docsearch = chroma_vectorize(documents)

    RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
    )

    print("Project repository selected: " + project_repository)
    print("Now, lets begin with the agent!")

    while True:
        question = input(
            "What would you like the agent to do? (type 'quit' or 'exit' to close)\n"
        )
        if question.lower() == "quit" or question.lower() == "exit":
            break

        OBJECTIVE = question

        # answer = chain({"question": OBJECTIVE})
        # print(answer["answer"])
        ask_agent(OBJECTIVE)


if __name__ == "__main__":
    main()
