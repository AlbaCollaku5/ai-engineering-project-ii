"""
Agentic AI Implementation
Part 5: Agent that decides between RAG and web search
"""
from vectorstore import vectorstore
from openrouter_integration import call_openrouter, is_openrouter_configured
import httpx
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_confidence_score(distances):
    """Convert distances to similarity scores"""
    return max([1 - d / 2 for d in distances]) if distances else 0

async def web_search(query: str) -> str:
    """Perform web search using DuckDuckGo API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("Abstract"):
                return f"Web search result: {data['Abstract']}"
            elif data.get("RelatedTopics"):
                first_topic = data["RelatedTopics"][0]
                if isinstance(first_topic, dict) and "Text" in first_topic:
                    return f"Web search result: {first_topic['Text']}"
            
            return "Web search completed but no relevant results found."
            
    except Exception as e:
        return f"Web search failed: {str(e)}"

async def agentic_query(q: str):
    """Agentic query with decision making"""
    # Get similar documents
    results = vectorstore.query_similar_docs(q, top_k=3)
    confidence = get_confidence_score(results['distances'])

    # Decision chain: if confidence < 0.5 → web search
    if confidence < 0.5:
        # Fallback to web search
        web_result = await web_search(q)
        return {
            "query": q, 
            "confidence": confidence, 
            "answer": web_result,
            "source": "web_search"
        }
    else:
        # Use RAG with retrieved documents
        docs = results['documents']
        prompt = "Context:\n" + "\n".join(docs) + f"\nQuestion: {q}"

        # Use OpenRouter for LLM response
        if is_openrouter_configured():
            answer = call_openrouter(prompt)
            source = "rag_openrouter"
        else:
            answer = "Set OpenRouter API key for AI generation."
            source = "rag_no_llm"

        return {
            "query": q, 
            "confidence": confidence, 
            "answer": answer,
            "source": source
        }