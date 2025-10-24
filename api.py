"""
FastAPI Deployment
Part 3: REST API with POST /documents and GET /query endpoints
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from vectorstore_safe import vectorstore
from agentic import agentic_query
import rag_pipeline_simple as rag_pipeline

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Knowledge Assistant", version="1.0.0")

class DocumentUpload(BaseModel):
    documents: list[str]

class QueryResponse(BaseModel):
    query: str
    confidence: float
    answer: str
    source: str

@app.post("/documents")
async def upload_documents(payload: DocumentUpload):
    """Upload and embed documents"""
    # Add to both vectorstore and RAG pipeline
    vectorstore.add_documents(payload.documents)
    rag_pipeline.simple_rag_pipeline.add_documents(payload.documents)
    return {"message": "Documents indexed successfully.", "count": len(payload.documents)}

@app.get("/query", response_model=QueryResponse)
async def query_endpoint(q: str):
    """Query with agentic decision making"""
    return await agentic_query(q)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Knowledge Assistant"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Knowledge Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /documents",
            "query": "GET /query?q=your_question",
            "health": "GET /health"
        }
    }