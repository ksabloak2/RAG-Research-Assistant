"""
Run this script once (or whenever you add new papers) to ingest PDFs into ChromaDB.

Usage:
    python ingest_papers.py
    python ingest_papers.py --papers-dir /path/to/pdfs
"""

import argparse
from src.ingest import run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Ingest research papers into ChromaDB.")
    parser.add_argument(
        "--papers-dir",
        default="./papers",
        help="Directory containing PDF files (default: ./papers)",
    )
    args = parser.parse_args()
    run_ingestion(papers_dir=args.papers_dir)


if __name__ == "__main__":
    main()
