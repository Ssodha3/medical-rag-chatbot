# rag_engine.py
import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

from typing import List
from langchain.schema import Document

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from sentence_transformers import CrossEncoder

# ===== FUNCTIONS =====

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def filter_to_minimal_documents(documents: List[Document]) -> List[Document]:
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        )
        for doc in documents
    ]

def text_split(minimal_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(minimal_docs)

def download_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ===== LOAD =====

print("Loading documents...")
docs = load_pdf_files("../Data")
minimal_docs = filter_to_minimal_documents(docs)
chunks = text_split(minimal_docs)
embedding = download_embeddings()

# ===== PINECONE =====

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "medicalchabot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding,
)

retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 6})

# ===== RERANKER =====

reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank_documents(query, docs, top_k=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked_docs[:top_k]]

# ===== LLM =====

chatModel = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

system_prompt = """
You are a helpful medical-information assistant, not a doctor.
Answer only from the retrieved context.
If the context is insufficient, say: "I don't have enough information in the retrieved documents to answer safely."
Do not invent treatments, diagnoses, or emergency instructions.
Keep the answer concise and include the source page if available.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answering_chain = create_stuff_documents_chain(chatModel, prompt)

# ===== FINAL FUNCTION =====

def ask_question(query):
    docs = retriever.invoke(query)
    reranked_docs = rerank_documents(query, docs, top_k=3)

    response = question_answering_chain.invoke({
        "input": query,
        "context": reranked_docs
    })

    if isinstance(response, dict):
        return response.get("answer", str(response))

    return response