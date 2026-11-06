from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

persistent_directory = "chromadb"

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

def retrival_section(query):

    retriever = db.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={
            "k": 5, 
            "score_threshold": 0.3
            }
        )
    # retriever = db.as_retriever(
    #     search_similarity="similarity_score_threshold",
    #     search_kwargs={
    #         "k": 5,
    #         "score_threshold":0.3
    #        }
    #     )

    relevant_docs = retriever.invoke(query)

    print(f"Query: {query}")
    print(f"Context")

    for i , doc in enumerate(relevant_docs,1):
        print(f"Document {i+1}: {doc.page_content}")

