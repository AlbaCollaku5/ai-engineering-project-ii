"""Simple MCP-like server exposing model context operations via HTTP (FastAPI)."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from orchestrator.orchestrator import orchestrate_query, add_documents

app = FastAPI(title="MCP Server (orchestrator)")


class QueryRequest(BaseModel):
    question: str
    summarize: Optional[bool] = True
    translate_to: Optional[str] = None


class DocsRequest(BaseModel):
    documents: list


@app.post("/query-docs")
def query_docs(req: QueryRequest):
    return orchestrate_query(req.question, summarize=req.summarize, translate_to=req.translate_to)


@app.post("/summarize")
def summarize(req: QueryRequest):
    return orchestrate_query(req.question, summarize=True)


@app.post("/translate")
def translate(req: QueryRequest):
    return orchestrate_query(req.question, summarize=False, translate_to=req.translate_to)


@app.post("/add-docs")
def add_docs(req: DocsRequest):
    count = add_documents(req.documents)
    return {"added": count}
