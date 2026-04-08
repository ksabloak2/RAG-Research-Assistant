# RAG Research Assistant

An end-to-end **Retrieval-Augmented Generation** pipeline for natural language Q&A over a corpus of AI/ML research papers. Drop in your PDFs, run one command to ingest, then ask questions in plain English.

## Architecture

```
papers/*.pdf
     │
     ▼
[PyPDF Loader] → [RecursiveCharacterTextSplitter] → [OpenAI Embeddings]
                                                            │
                                                            ▼
                                                      [ChromaDB]
                                                            │
                  user query ──────────────────────> [Retriever (top-k)]
                                                            │
                                                            ▼
                                                    [GPT-4o-mini LLM]
                                                            │
                                                            ▼
                                               context-aware answer + sources
```

**Stack:** Python · LangChain · ChromaDB · OpenAI API (`text-embedding-3-small` + `gpt-4o-mini`)

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/ksabloak/RAG-Research-Assistant.git
cd RAG-Research-Assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add research papers

Place PDF files in the `papers/` directory:

```
papers/
  attention_is_all_you_need.pdf
  bert.pdf
  gpt3.pdf
  ...
```

### 4. Ingest papers

```bash
python ingest_papers.py
```

This loads, chunks (1 000 tokens, 200 overlap), embeds, and persists all PDFs into ChromaDB.

### 5. Ask questions

**Interactive REPL:**
```bash
python main.py
```

**Single query:**
```bash
python main.py --query "How does multi-head attention work?"
```

**Re-ingest + query:**
```bash
python main.py --reingest
```

## Example session

```
=== RAG Research Assistant ===
Ask questions about the indexed AI/ML papers. Type 'exit' to stop.

You: What is the key innovation of the Transformer architecture?
Assistant: The Transformer replaces recurrence entirely with self-attention mechanisms,
allowing the model to draw global dependencies between input and output regardless of
distance. (Source: attention_is_all_you_need.pdf)
Sources: attention_is_all_you_need.pdf

You: How does BERT differ from GPT in pre-training?
Assistant: BERT uses masked language modeling (MLM) and next-sentence prediction for
bidirectional pre-training, while GPT uses causal (left-to-right) language modeling.
Sources: bert.pdf, gpt3.pdf
```

## Project structure

```
RAG-Research-Assistant/
├── src/
│   ├── config.py        # All settings (env-backed)
│   ├── ingest.py        # PDF loading, chunking, embedding → ChromaDB
│   └── rag_chain.py     # LangChain RetrievalQA chain
├── tests/
│   ├── test_ingest.py
│   └── test_rag_chain.py
├── papers/              # Add your PDFs here (gitignored)
├── chroma_db/           # Auto-created on first ingest (gitignored)
├── ingest_papers.py     # Ingestion entry point
├── main.py              # Q&A entry point
├── requirements.txt
└── .env.example
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI key. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `PAPERS_DIR` | `./papers` | PDF source directory |
| `CHUNK_SIZE` | `1000` | Tokens per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

## Running tests

```bash
pip install pytest
pytest tests/
```
