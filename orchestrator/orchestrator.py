import time
import os
from typing import Dict, Any, Optional

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project imports
import rag_pipeline_simple as rag_pipeline
from prompts.loader import load_prompt
from monitoring.advanced_monitoring import advanced_metrics
from a2a_client import a2a_client
import asyncio


metrics = advanced_metrics


def summarizer(text: str, prompt_name: str = "summarize_v1") -> str:
    """Simple summarizer stub that uses prompt templates from registry."""
    prompt = load_prompt(prompt_name)
    # Very small stub: include prompt and first 500 chars
    return f"SUMMARY (using {prompt['name']}): {text[:500]}"


def translator(text: str, target_language: str = "en") -> str:
    """Simple translator stub (placeholder)."""
    return f"[Translated to {target_language}] {text}"


def orchestrate_query(question: str, summarize: bool = True, translate_to: Optional[str] = None, use_a2a: bool = False) -> Dict[str, Any]:
    """Run RAG -> summarizer -> translator chain and collect monitoring metrics.

    Inputs: question (str), summarize (bool), translate_to (optional language code)
    Outputs: dict with answer, sources, runtime_ms, retrieval_confidence
    """
    start = time.time()
    # Query RAG
    rag_start = time.time()
    rag_result = rag_pipeline.simple_rag_pipeline.query(question)
    rag_latency = (time.time() - rag_start) * 1000

    answer = rag_result.get("answer", "")
    sources = rag_result.get("source_documents", [])

    # Optionally summarize
    if summarize:
        sum_start = time.time()
        answer = summarizer(answer)
        metrics.record_metric("summarization_latency_ms", (time.time() - sum_start) * 1000)

    # Optionally translate
    if translate_to:
        trans_start = time.time()
        if use_a2a:
            # Use A2A translation service
            try:
                # Create new event loop for A2A call
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                translation_result = loop.run_until_complete(a2a_client.call_translation_service(answer, translate_to))
                loop.close()
                
                if "translation" in translation_result:
                    answer = translation_result["translation"]
                else:
                    answer = translator(answer, translate_to)  # Fallback
            except Exception as e:
                print(f"A2A translation failed: {e}")
                answer = translator(answer, translate_to)  # Fallback
        else:
            answer = translator(answer, translate_to)
        metrics.record_metric("translation_latency_ms", (time.time() - trans_start) * 1000)

    total_ms = (time.time() - start) * 1000

    # Confidence from RAG
    confidence = rag_pipeline.simple_rag_pipeline.get_confidence_score(question)

    # Record metrics
    metrics.record_metric("retrieval_latency_ms", rag_latency)
    metrics.record_metric("total_request_ms", total_ms)
    metrics.record_metric("retrieval_confidence", confidence)

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_confidence": confidence,
        "metrics": metrics.latest_snapshot()
    }


def add_documents(documents):
    """Helper to add documents into the underlying RAG pipeline/vectorstore."""
    return rag_pipeline.simple_rag_pipeline.add_documents(documents)
