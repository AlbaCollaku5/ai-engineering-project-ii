# Project 2 – AI Orchestrator & Monitoring Platform

## Objective
Build on **Project 1** by extending the assistant into a **production-ready AI platform**. Add orchestration, monitoring, protocols (MCP, A2A), fine-tuning, and evaluation. This project assumes **Project 1 is completed**.

---

## Requirements

### Part 1: LLM Orchestration
- Implement orchestration via **LangChain Expression Language** or **Haystack pipelines**.
- Chain multiple tools:
  - RAG from Project 1
  - Summarizer
  - Translation tool

### Part 2: Model Evaluation & Monitoring
- Integrate **Evidently AI** or **WhyLabs** for monitoring.
- Track metrics: latency, token usage, retrieval accuracy.
- Store evaluations in a DB (e.g., Postgres).

### Part 3: MCP Server Protocol
- Implement **Model Context Protocol** to make the assistant accessible by external apps (VS Code, CLI).
- Register commands: `query-docs`, `summarize`, `translate`.

### Part 4: A2A Protocol
- Create an **API-to-API integration**.
- Example: Your assistant queries another AI service for sentiment analysis.
- Implement a REST client inside the orchestrator.

### Part 5: Prompt Engineering
- Maintain a **prompt registry** (YAML/JSON).
- Version prompts and measure performance.
- Include at least 3 optimized prompt templates for different tasks.

### Part 6: Fine-Tuning & PEFT
- Fine-tune or use **PEFT adapters** on a smaller LLM (e.g., LoRA on LLaMA-2 or FLAN-T5).
- Integrate fine-tuned model as an optional backend in the orchestrator.

---

## Deliverables
- Extended repo (building on Project 1)
- Monitoring dashboards (Streamlit or Grafana)
- MCP server integration
- Documentation of prompts & tuning
- End-to-end demo script

---

## Evaluation Criteria
| Category | Points |
|----------|--------|
LLM Orchestration | 20 |
Monitoring & Evaluation | 20 |
MCP Server Protocol | 15 |
A2A Protocol Integration | 10 |
Prompt Engineering | 10 |
Fine-Tuning & PEFT | 25 |
**Total** | **100 pts** |

---

## Milestones
1. Orchestration + monitoring (Week 1–2)  
2. MCP + A2A protocols (Week 3)  
3. Prompt engineering + fine-tuning (Week 4–5)  
4. Demo + docs (Week 6)  

---
