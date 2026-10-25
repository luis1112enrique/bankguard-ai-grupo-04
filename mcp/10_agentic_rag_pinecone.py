"""
10_agentic_rag_pinecone.py

Agentic RAG over a persistent Pinecone student-data-governance knowledge base.

At runtime the agent does NOT reopen the local source files.
It decides when to search Pinecone and uses the retrieved evidence to answer.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

# Keep this local course example quiet unless tracing is explicitly requested.
os.environ["LANGSMITH_TRACING"] = os.getenv("AGENTIC_RAG_TRACING", "false")

INDEX_NAME = os.environ["INDEX_NAME"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-large")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
)


@tool
def search_governance_knowledge(query: str) -> str:
    """Search student-data definitions, privacy documents, AI rules, ownership, and retention policies."""

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant governance evidence was found."

    evidence = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        row = doc.metadata.get("row")

        location = source
        if page is not None:
            location += f", page {page}"
        if row is not None:
            location += f", row {row}"

        evidence.append(f"SOURCE: {location}\n" f"{doc.page_content}")

    return "\n\n---\n\n".join(evidence)


llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0,
    reasoning=False,
)

agent = create_agent(
    model=llm,
    tools=[search_governance_knowledge],
    system_prompt="""
You are a Data Steward for student data.

Use the governance retrieval tool whenever a question depends on:
- privacy requirements
- data classification
- data ownership
- approved uses
- retention
- AI or analytics use of student data

Base governance claims only on retrieved evidence.
Distinguish between official/public documents and synthetic teaching policies
when the source information makes that distinction clear.

If the retrieved evidence is insufficient, conflicting, or does not authorize
the proposed use, say so. Do not invent permission.

Answer only student-data-governance questions. If all or part of a request is
unrelated, explain briefly that part is outside your role instead of presenting
an unsupported opinion.

When possible, mention the source filename that supports the answer.
""",
)


def print_final_text(result):
    message = result["messages"][-1]

    if isinstance(message.content, str):
        print(message.content)
        return

    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""))


def ask_question(question: str, conversation: list) -> list:
    """Run one question through the agent and print its final answer."""
    result = agent.invoke(
        {
            "messages": [
                *conversation,
                {
                    "role": "user",
                    "content": question,
                },
            ]
        }
    )

    print("\nANSWER:")
    print_final_text(result)
    return result["messages"]


def main() -> None:
    """Run an interactive governance question-and-answer loop."""
    print("Student Data Governance Agent")
    print("Ask a question, or type 'salir' to finish.")
    conversation = []

    while True:
        try:
            question = input("\nQUESTION: ").strip()
        except EOFError:
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.casefold() in {"exit", "quit", "salir"}:
            print("Goodbye.")
            return

        try:
            conversation = ask_question(question, conversation)
        except Exception as error:
            print(f"\nERROR: {error}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
