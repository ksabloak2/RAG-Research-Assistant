"""
Interactive Q&A interface for the RAG Research Assistant.

Usage:
    python main.py                  # interactive REPL
    python main.py --query "..."    # single question, then exit
    python main.py --reingest       # re-ingest papers before starting
"""

import argparse
import os
import sys

from src.config import CHROMA_PERSIST_DIR
from src.ingest import run_ingestion, load_vectorstore
from src.rag_chain import build_rag_chain, query


def get_vectorstore(reingest: bool, papers_dir: str):
    if reingest or not os.path.exists(CHROMA_PERSIST_DIR):
        return run_ingestion(papers_dir=papers_dir)
    print(f"Loading existing ChromaDB index from '{CHROMA_PERSIST_DIR}'...")
    return load_vectorstore()


def interactive_loop(chain):
    print("\n=== RAG Research Assistant ===")
    print("Ask questions about the indexed AI/ML papers. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = query(chain, question)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print()


def main():
    parser = argparse.ArgumentParser(description="RAG Research Assistant")
    parser.add_argument("--query", "-q", help="Single question to answer (non-interactive mode)")
    parser.add_argument("--reingest", action="store_true", help="Re-ingest all papers before querying")
    parser.add_argument("--papers-dir", default="./papers", help="PDF directory for ingestion")
    args = parser.parse_args()

    vectorstore = get_vectorstore(reingest=args.reingest, papers_dir=args.papers_dir)
    chain = build_rag_chain(vectorstore)

    if args.query:
        result = query(chain, args.query)
        print(f"Answer: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
    else:
        interactive_loop(chain)


if __name__ == "__main__":
    main()
