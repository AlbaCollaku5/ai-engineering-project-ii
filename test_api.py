#!/usr/bin/env python3
"""
Simple test script for the AI Knowledge Assistant API
"""
import urllib.request
import urllib.parse
import json

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing AI Knowledge Assistant API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        with urllib.request.urlopen(f"{base_url}/health") as response:
            data = json.loads(response.read().decode())
            print(f"✅ Health check: {data}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        with urllib.request.urlopen(f"{base_url}/") as response:
            data = json.loads(response.read().decode())
            print(f"✅ Root endpoint: {data}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test 3: Upload documents
    print("\n3. Testing document upload...")
    try:
        documents = {
            "documents": [
                "Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers.",
                "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently.",
                "ChromaDB is an open-source vector database that makes it easy to build AI applications with embeddings."
            ]
        }
        
        data = json.dumps(documents).encode('utf-8')
        req = urllib.request.Request(
            f"{base_url}/documents",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"✅ Document upload: {result}")
    except Exception as e:
        print(f"❌ Document upload failed: {e}")
    
    # Test 4: Query with high confidence (should use RAG)
    print("\n4. Testing query with high confidence (RAG)...")
    try:
        query = "What is Docker?"
        encoded_query = urllib.parse.quote(query)
        with urllib.request.urlopen(f"{base_url}/query?q={encoded_query}") as response:
            data = json.loads(response.read().decode())
            print(f"✅ RAG Query: {data}")
    except Exception as e:
        print(f"❌ RAG Query failed: {e}")
    
    # Test 5: Query with low confidence (should use web search)
    print("\n5. Testing query with low confidence (Web Search)...")
    try:
        query = "What is blockchain technology?"
        encoded_query = urllib.parse.quote(query)
        with urllib.request.urlopen(f"{base_url}/query?q={encoded_query}") as response:
            data = json.loads(response.read().decode())
            print(f"✅ Web Search Query: {data}")
    except Exception as e:
        print(f"❌ Web Search Query failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")

if __name__ == "__main__":
    test_api()
