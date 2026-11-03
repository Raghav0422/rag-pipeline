from langchain_core import vectorstores
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DOCS_PATH = os.path.join(PROJECT_ROOT, "database")

def create_vector_store(chunks,persist_directory="chromadb"):
    print("creating embedding and storing in chroma db")

    # try:
    #     embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    # except Exception as e:
    #     print("open ai embedding failed")
    #     embedding_model = HuggingFaceEmbeddings(
    #         model_name="sentence-transformers/all-MiniLM-L6-v2"
    #     )
    
    embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    print("creating vector store")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space" : "cosine"}
    )
    print("vector store is created fgor the chroma DB")

    return vectorstore


    