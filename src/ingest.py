"""
Document ingestion pipeline: load PDFs → chunk → embed → store in ChromaDB.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import (
    OPENAI_API_KEY,
    CHROMA_PERSIST_DIR,
    PAPERS_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)


def load_pdfs(papers_dir: str) -> list:
    """Load all PDFs from the papers directory."""
    papers_path = Path(papers_dir)
    pdf_files = list(papers_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{papers_dir}'. Add research papers there.")

    documents = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        # Tag each chunk with its source filename for traceability
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        documents.extend(docs)

    print(f"Loaded {len(documents)} pages from {len(pdf_files)} papers.")
    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def embed_and_store(chunks: list) -> Chroma:
    """Embed chunks with OpenAI and persist to ChromaDB."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()
    print(f"Stored embeddings in ChromaDB at '{CHROMA_PERSIST_DIR}' (collection: {COLLECTION_NAME}).")
    return vectorstore


def run_ingestion(papers_dir: str = PAPERS_DIR) -> Chroma:
    """Full ingestion pipeline: PDF → chunks → ChromaDB."""
    print("=== Ingestion Pipeline ===")
    documents = load_pdfs(papers_dir)
    chunks = chunk_documents(documents)
    vectorstore = embed_and_store(chunks)
    print("Ingestion complete.\n")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB collection (no re-ingestion)."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
