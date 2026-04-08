"""Unit tests for the RAG chain (mocked LLM and vectorstore)."""

from unittest.mock import MagicMock, patch
from langchain.schema import Document


@patch("src.rag_chain.ChatOpenAI")
@patch("src.rag_chain.RetrievalQA")
def test_query_returns_answer_and_sources(mock_retrieval_qa_cls, mock_llm_cls):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "Self-attention allows each token to attend to all others.",
        "source_documents": [
            Document(page_content="...", metadata={"source": "attention.pdf"}),
            Document(page_content="...", metadata={"source": "attention.pdf"}),
        ],
    }
    mock_retrieval_qa_cls.from_chain_type.return_value = mock_chain

    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()

    from src.rag_chain import build_rag_chain, query
    chain = build_rag_chain(mock_vectorstore)
    result = query(chain, "What is self-attention?")

    assert "self-attention" in result["answer"].lower()
    assert result["sources"] == ["attention.pdf"]
