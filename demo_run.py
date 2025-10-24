"""Small demo runner showcasing orchestrator and MCP server usage."""
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from orchestrator.orchestrator import orchestrate_query, add_documents


def run_demo():
    print("Adding sample documents to vectorstore...")
    docs = [
        "This is a sample document about AI engineering and orchestration.",
        "Another doc covering monitoring and evaluation practices."
    ]
    added = add_documents(docs)
    print(f"Added {added} chunks to vectorstore")

    print("Running a sample query through the orchestrator...")
    start = time.time()
    res = orchestrate_query("What is AI orchestration?", summarize=True)
    print("Result:")
    print(res)
    print(f"Elapsed: {(time.time()-start)*1000:.2f} ms")


if __name__ == '__main__':
    run_demo()
