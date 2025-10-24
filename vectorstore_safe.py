"""
Safe Vector Database & Embeddings Implementation
Handles ChromaDB initialization errors gracefully
"""
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

class SafeVectorStore:
    """ChromaDB vector store with error handling"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.embedding_model = None
        self.collection = None
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get or create collection with error handling
            try:
                self.collection = self.client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                print("SUCCESS: ChromaDB initialized successfully")
            except Exception as e:
                print(f"ERROR: ChromaDB collection error: {e}")
                print("FIX: Attempting to fix database...")
                self._fix_database()
                
        except Exception as e:
            print(f"ERROR: ChromaDB initialization failed: {e}")
            print("FIX: Please run: python fix_chromadb.py")
    
    def _fix_database(self):
        """Try to fix database issues"""
        try:
            # Remove corrupted database
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)
            
            # Reinitialize
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            print("SUCCESS: Database fixed and reinitialized")
        except Exception as e:
            print(f"ERROR: Failed to fix database: {e}")
    
    def add_documents(self, documents: List[str]) -> int:
        """Add documents to vector store"""
        if not self.collection or not self.embedding_model:
            print("ERROR: ChromaDB not properly initialized")
            return 0
            
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Generate IDs
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids
            )
            
            return len(documents)
        except Exception as e:
            print(f"Error adding documents: {e}")
            return 0
    
    def query_similar_docs(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Query similar documents"""
        if not self.collection or not self.embedding_model:
            print("ERROR: ChromaDB not properly initialized")
            return {'documents': [], 'distances': [], 'ids': []}
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            return {
                'documents': results['documents'][0] if results['documents'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'ids': results['ids'][0] if results['ids'] else []
            }
        except Exception as e:
            print(f"Error querying documents: {e}")
            return {'documents': [], 'distances': [], 'ids': []}

# Global instance
vectorstore = SafeVectorStore()
