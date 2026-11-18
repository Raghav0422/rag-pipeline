from src.ingestion.ingestion_pipeline import load_documents
from src.chunking.flat_chunking import split_documents
from src.embedding.openai import create_vector_store, creating_embedding_model

from fastapi import FastAPI
from fastapi.routing import APIRouter
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from langchain_huggingface import HuggingFaceEmbeddings
from api.routes import router

from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Creating embedding model for the app state")

    app.state.embedding_model = await creating_embedding_model()
    app.state.vector_store = create_vector_store(app.state.embedding_model)
    # main()
    print(app.state)    
    print("ruuning the app now")
    yield

    app.state.embedding_model = None
    app.state.vector_store = None

app = FastAPI(lifespan = lifespan)
app.include_router(router, prefix="/rag")

# import os
# APP_MODE = os.getenv("APP_MODE", "query")
# def main():
#     print("Main function started")
#     if APP_MODE == "ingest":
        

# if __name__ == "__main__":
#     main()