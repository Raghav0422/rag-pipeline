import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DOCS_PATH = os.path.join(PROJECT_ROOT, "docs")


def load_documents(docs_path=None):
    docs_path = os.path.abspath(docs_path or DEFAULT_DOCS_PATH)
    print(f"Loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents path {docs_path} does not exist")

    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True,
        },
    )

    document = loader.load()

    if len(document) == 0:
        raise FileNotFoundError(f"No documents found in {docs_path}")

    for i , doc in enumerate(document):
        print(f"Document {i+1} loaded successfully")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content LEngth : {len(doc.page_content)} characters")
        print(f"Content Preview : {doc.page_content[:100]}...")
        print(f"metadata : {doc.metadata}")
        
    # print(document)

    return document
    

