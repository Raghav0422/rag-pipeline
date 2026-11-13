import argparse

from src.ingestion.ingestion_pipeline import load_documents
from src.chunking.flat_chunking import split_documents
from src.embedding.openai import create_vector_store
from src.retrieval.retrieval import retrival_section

def main():
    print("Main function started")

    parser = argparse.ArgumentParser(description="Run ingestion or retrieval.")
    parser.add_argument(
        "--mode",
        choices=["ingestion", "retrieval"],
        required=True,
        help="Operation to run",
    )
    parser.add_argument(
        "--query",
        default="in which year microsoft was build?",
        help="Question to use in retrieval mode",
    )
    args = parser.parse_args()

    if args.mode == "ingestion":
        print("Calling Ingestion pipeline")
        documents = load_documents()
        print("Calling Chunking pipeline")
        chunks = split_documents(documents)
        print("Calling creation for vector store")
        vectorestore = create_vector_store(chunks)
    elif args.mode == "retrieval":
        query = args.query or " when google was founded?"
        relevant_docs = retrival_section(query)
        from src.llm.llm import get_llm
        llm_response = get_llm(query, relevant_docs)
    else:
        print("Invalid mode. Please specify 'ingest' or 'query'.")

if __name__ == "__main__":
    main()