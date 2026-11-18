from fastapi import APIRouter, HTTPException,Request
from fastapi.responses import JSONResponse

from src.chunking.flat_chunking import split_documents
from src.embedding.openai import create_vector_store
from src.ingestion.ingestion_pipeline import load_documents

router = APIRouter(prefix="/v1")

@router.get("/health")
async def health_check(request : Request):
    embedding_model = getattr(request.app.state, "embedding_model", None) is not None

    if embedding_model:
        return {"status": "healthy", 
                "models":{
                    "embedding_model" : embedding_model
                } 
        }

    return JSONResponse(status_code=503, content={
        "status": "unhealthy", 
        "models":{
            "embedding_model" : "loadded"  if embedding_model else "not loaded"
        }
        })

@router.get("/ingest")
async def ingest_data(request : Request):
    vector_store = request.app.state.vector_store
    documents = load_documents()
    print("Document loading completed")
    chunks = split_documents(documents)
    print("Chunking completed")

    ids = []
    if chunks:
        ids.extend(
            f"{chunks.metadata['source']}:{index}" for index, chunks in enumerate(chunks)
            )

    vector_store.add_documents(chunks,ids=list(ids))

    if vector_store:
        return {
            "status": "success",
            "message": "Document loading and chunking comleted",
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "vector_store": {
                "collection_name": vector_store._collection.name,
                "count": vector_store._collection.count(),
                "metadata": vector_store._collection.metadata
            }            
        }

@router.get("/inspect")
async def inspect(request : Request):
    embedding_model = request.app.state.embedding_model
    vector_store = request.app.state.vector_store

    collection = vector_store._collection

    result = collection.get(
        include=["documents", "metadatas"]
    )

    count = collection.count()
    collection_name = collection.name
    collection_metadata = collection.metadata
    chunks = [
        {
            "id": chunk_id,
            "metadata": metadata,
            "text": document,
        }
        for chunk_id, document, metadata in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        )
    ]
    collection_info = {
        "embedding_model": embedding_model,
        "count": count,
        "collection_name": collection_name,
        "metadata": collection_metadata,
        "chunks": chunks
    }

    return {"status": "success", "collection_info": collection_info}