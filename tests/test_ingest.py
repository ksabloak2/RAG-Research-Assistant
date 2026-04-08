"""Unit tests for the ingestion pipeline (no OpenAI calls — mocked)."""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document


@patch("src.ingest.PyPDFLoader")
def test_load_pdfs_tags_source(mock_loader_cls, tmp_path):
    # Create a fake PDF file
    fake_pdf = tmp_path / "attention.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    mock_doc = Document(page_content="Attention is all you need.", metadata={})
    mock_loader = MagicMock()
    mock_loader.load.return_value = [mock_doc]
    mock_loader_cls.return_value = mock_loader

    from src.ingest import load_pdfs
    docs = load_pdfs(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "attention.pdf"


def test_chunk_documents_produces_chunks():
    from src.ingest import chunk_documents
    doc = Document(page_content="word " * 500, metadata={"source": "test.pdf"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1


def test_load_pdfs_raises_when_empty(tmp_path):
    from src.ingest import load_pdfs
    with pytest.raises(FileNotFoundError):
        load_pdfs(str(tmp_path))
