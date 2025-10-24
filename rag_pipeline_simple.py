"""
Simplified RAG Pipeline Implementation
Uses only basic similarity search without complex chains
"""
import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openrouter_integration import call_openrouter, is_openrouter_configured


class SimpleRAGPipeline:
    """Simplified RAG pipeline using basic similarity search"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def add_documents(self, documents: List[str]) -> int:
        """Add documents to the vector store"""
        try:
            # Split documents into chunks
            texts = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc)
                texts.extend(chunks)
            
            # Add to vector store
            self.vectorstore.add_texts(texts)
            
            return len(texts)
        except Exception as e:
            print(f"Failed to add documents: {e}")
            return 0
    
    def query(self, question: str, use_llm: bool = True) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        try:
            # Get relevant documents
            docs = self.vectorstore.similarity_search(question, k=3)
            
            if use_llm and is_openrouter_configured():
                # Use OpenRouter for AI generation
                context = "\n".join([doc.page_content for doc in docs])
                prompt = f"""Use the following context to answer the question. If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
                
                # For now, just return a placeholder since we don't have OpenRouter configured
                answer = "This is a placeholder answer. Configure OpenRouter API key for AI generation."
                return {
                    "answer": answer,
                    "source_documents": [doc.page_content for doc in docs],
                    "method": "openrouter_rag"
                }
            else:
                # Return context without AI generation
                return {
                    "answer": "No LLM configured. Set OpenRouter API key for AI generation.",
                    "source_documents": [doc.page_content for doc in docs],
                    "method": "similarity_search"
                }
        except Exception as e:
            print(f"Query failed: {e}")
            return {
                "answer": f"Query failed: {str(e)}",
                "source_documents": [],
                "method": "error"
            }
    
    def get_confidence_score(self, question: str) -> float:
        """Get confidence score for a query"""
        try:
            docs = self.vectorstore.similarity_search_with_score(question, k=3)
            if docs:
                # Convert distances to similarity scores
                scores = [1 - score for _, score in docs]
                return max(scores)
            return 0.0
        except Exception as e:
            print(f"Confidence score calculation failed: {e}")
            return 0.0


# Global instance
simple_rag_pipeline = SimpleRAGPipeline()
