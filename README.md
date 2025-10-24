<<<<<<< HEAD
=======
# ai-engineering-project-ii

>>>>>>> becb600a378acfa330ac5ee2ffcf0f286bf4c001
# ai-engineering-part-ii
This repository contains the Project 1 baseline and new scaffolding for Project 2: AI Orchestrator & Monitoring Platform.

Key additions (Part II):
- Orchestrator chaining RAG -> summarizer -> translator (`orchestrator/orchestrator.py`).
- Simple monitoring collector persisting metrics to SQLite (`monitoring/`).
- MCP-like FastAPI server exposing `query-docs`, `summarize`, `translate` (`mcp_server.py`).
- A2A client example for API-to-API calls (`a2a_client.py`).
- Prompt registry with YAML templates and loader (`prompts/`).
- Fine-tuning/PEFT placeholder and instructions (`fine_tuning/`).
- Demo runner and examples (`demo_run.py`).

See `ai-engineering-part-ii.md` for the full project brief and milestones.

How to run (local):

1. Install requirements (recommended virtualenv).

```powershell
python -m pip install -r requirements.txt
```

2. Run demo locally:

```powershell
python demo_run.py
```

3. Run MCP server (FastAPI) with uvicorn:

```powershell
uvicorn mcp_server:app --reload --port 8000
```

This README will be expanded with dashboards and evaluation integration in future commits.
# ai-engineering-part-i
The AI Engineering Part I Project - LHIND Internal Training
