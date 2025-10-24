#!/usr/bin/env python3
"""
Fix ChromaDB database corruption
"""
import os
import shutil

def fix_chromadb():
    """Remove corrupted ChromaDB files and create fresh database"""
    
    # Path to chroma_db directory
    chroma_path = "./chroma_db"
    
    if os.path.exists(chroma_path):
        print(f"Found existing ChromaDB at {chroma_path}")
        print("Removing corrupted database...")
        
        try:
            shutil.rmtree(chroma_path)
            print("✅ Removed corrupted ChromaDB database")
        except Exception as e:
            print(f"❌ Error removing database: {e}")
            return False
    else:
        print("No existing ChromaDB database found")
    
    # Create fresh directory
    try:
        os.makedirs(chroma_path, exist_ok=True)
        print("✅ Created fresh ChromaDB directory")
        return True
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing ChromaDB database corruption...")
    if fix_chromadb():
        print("🎉 ChromaDB database fixed! You can now run the application.")
    else:
        print("❌ Failed to fix ChromaDB database.")
