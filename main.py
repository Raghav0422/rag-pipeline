from src.ingestion.ingestion_pipeline import load_documents


def main():
    print("Main function started")
    print("Calling Ingestion pipeline")
    load_documents()

if __name__ == "__main__":
    main()