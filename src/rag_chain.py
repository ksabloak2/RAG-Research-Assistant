"""
LangChain retrieval chain: ChromaDB retriever + OpenAI LLM + prompt template.
"""

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from src.config import OPENAI_API_KEY, LLM_MODEL, RETRIEVER_K

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a research assistant with deep expertise in AI and machine learning.
Use the retrieved excerpts below to answer the question precisely and concisely.
Cite the source paper (filename) when referencing specific claims.
If the answer is not contained in the context, say "I don't have enough information in the indexed papers."

Context:
{context}

Question: {question}

Answer:""",
)


def build_rag_chain(vectorstore: Chroma) -> RetrievalQA:
    """Construct a RetrievalQA chain from the vectorstore."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": SYSTEM_PROMPT},
    )
    return chain


def query(chain: RetrievalQA, question: str) -> dict:
    """Run a question through the RAG chain and return answer + sources."""
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = sorted({
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    })
    return {"answer": answer, "sources": sources}
