from src.ingestion.ingestion_pipeline import load_documents
from src.chunking.flat_chunking import split_documents
from src.embedding.openai import create_vector_store

def main():
    print("Main function started")
    print("Calling Ingestion pipeline")
    documents = load_documents()
    print("Calling Chunking pipeline")
    chunks = split_documents(documents)
    print("Calling creation for vector store")
    vectorestore = create_vector_store(chunks)

if __name__ == "__main__":
    main()