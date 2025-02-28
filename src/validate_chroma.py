import os
import chromadb

# Define ChromaDB storage path (same as in rag_poc.py & weavr.py)
persist_directory = r"C:\Users\hoggw\Documents\weavr-alpha\db\vector_store"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=persist_directory)

# Print debug info
print(f"Using ChromaDB at: {persist_directory}")
print("Available collections:", client.list_collections())

# Attempt to retrieve stored documents
try:
    collection = client.get_collection("weavr_knowledge_base")
    print(f"Document count: {collection.count()}")
except Exception as e:
    print(f"Error retrieving collection: {e}")
