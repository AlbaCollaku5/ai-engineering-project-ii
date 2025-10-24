"""
Python Client SDK
Part 4: Easy-to-use client library for the AI Knowledge Assistant API
"""
import httpx
import asyncio
from typing import List, Dict, Any

class AIKnowledgeClient:
    """Async client for AI Knowledge Assistant API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def upload_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Upload documents to the knowledge base"""
        response = await self.client.post(
            f"{self.base_url}/documents",
            json={"documents": documents}
        )
        response.raise_for_status()
        return response.json()
    
    async def query(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base"""
        response = await self.client.get(
            f"{self.base_url}/query",
            params={"q": question}
        )
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()

class AIKnowledgeClientSync:
    """Synchronous client for AI Knowledge Assistant API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client()
    
    def upload_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Upload documents to the knowledge base"""
        response = self.client.post(
            f"{self.base_url}/documents",
            json={"documents": documents}
        )
        response.raise_for_status()
        return response.json()
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base"""
        response = self.client.get(
            f"{self.base_url}/query",
            params={"q": question}
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the client"""
        self.client.close()

# Example usage
if __name__ == "__main__":
    # Async usage
    async def async_example():
        client = AIKnowledgeClient()
        
        # Upload documents
        result = await client.upload_documents([
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand human language."
        ])
        print("Upload result:", result)
        
        # Query the knowledge base
        response = await client.query("What is machine learning?")
        print("Query response:", response)
        
        await client.close()
    
    # Sync usage
    def sync_example():
        client = AIKnowledgeClientSync()
        
        # Upload documents
        result = client.upload_documents([
            "Python is a programming language.",
            "FastAPI is a modern web framework for Python."
        ])
        print("Upload result:", result)
        
        # Query the knowledge base
        response = client.query("What is Python?")
        print("Query response:", response)
        
        client.close()
    
    # Run examples
    print("Running async example...")
    asyncio.run(async_example())
    
    print("\nRunning sync example...")
    sync_example()