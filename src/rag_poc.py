import os
import time
import shutil
import markdown
import requests
import hashlib
import sys
from bs4 import BeautifulSoup
from collections import defaultdict
from pathlib import Path
from typing import List
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_together import TogetherEmbeddings

# Ensure the main project directory is in the Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config_loader import load_api_key
from together import Together

# Configuration
TOGETHER_API_KEY = load_api_key()
if not TOGETHER_API_KEY:
    raise ValueError("ERROR: TOGETHER_API_KEY is missing from config.yaml!")

# Initialize API Client
embedding_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=TOGETHER_API_KEY)
client = Together(api_key=TOGETHER_API_KEY)

# Paths
HOME_DIR = Path(os.path.expanduser("~"))
WEAVR_DIR = HOME_DIR / "Documents" / "weavr-alpha"
FAISS_INDEX_PATH = WEAVR_DIR / "data" / "faiss_index"
OBS_VAULT_PATH = HOME_DIR / "Documents" / "The Scriptorium"
KNOWLEDGE_BASE_PATH = OBS_VAULT_PATH / "weavr_knowledge_base"

if not KNOWLEDGE_BASE_PATH.exists():
    raise ValueError(f"Obsidian vault not found at: {KNOWLEDGE_BASE_PATH}")

# FAISS Expiration Settings
EXPIRATION_DAYS = 30

def is_faiss_expired():
    """Checks if FAISS index is older than 30 days."""
    if not FAISS_INDEX_PATH.exists():
        return True  # No index exists, so it's "expired" by default
    
    creation_time = os.path.getctime(FAISS_INDEX_PATH)
    age_days = (time.time() - creation_time) / (60 * 60 * 24)
    return age_days > EXPIRATION_DAYS

def delete_faiss_index():
    """Deletes FAISS index directory if it exists."""
    if FAISS_INDEX_PATH.exists():
        print("🔹 FAISS index is expired. Deleting old index...")
        shutil.rmtree(FAISS_INDEX_PATH)

def load_or_create_faiss(chunks, embedding_model):
    """Loads FAISS index if it exists and is valid; otherwise, creates a new one."""
    if is_faiss_expired():
        delete_faiss_index()
    
    if FAISS_INDEX_PATH.exists():
        print("🔹 Loading existing FAISS index...")
        try:
            return FAISS.load_local(str(FAISS_INDEX_PATH), embedding_model)
        except Exception as e:
            print(f"⚠️ FAISS index corrupt or incompatible: {e}. Rebuilding index.")
            delete_faiss_index()
    
    print("🔹 Creating new FAISS index...")
    vectorstore = FAISS.from_documents(unique_chunks, embedding_model, metadatas=[{"source": chunk.metadata["source"]} for chunk in unique_chunks])
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    return vectorstore

def parse_markdown_file(filepath):
    """Parses a Markdown file and extracts text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    html = markdown.markdown(markdown_content)
    text = ''.join(BeautifulSoup(html, "html.parser").find_all(string=True))
    return text

def load_documents():
    """Dynamically loads all Markdown files from the knowledge base."""
    documents = []
    for filepath in KNOWLEDGE_BASE_PATH.rglob("*.md"):
        try:
            text_content = parse_markdown_file(filepath)
            print(f"{filepath.name} parsed. Length: {len(text_content)}")
            documents.append(Document(page_content=text_content, metadata={"source": filepath.name}))
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    return documents

# Load and Process Documents
documents = load_documents()
if documents:
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase chunk size slightly
    chunk_overlap=200,  # Keep more context in each chunk
    separators=["\n\n", "\n", " ", ""],  # Prioritize splitting on paragraph breaks
)
    all_chunks = text_splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(all_chunks)}")

    # Deduplication
    content_hashes = defaultdict(int)
    
    def deduplicate_chunks(chunks):
        """Removes duplicate document chunks based on hash values."""
        unique_chunks = []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
            if content_hashes[chunk_hash] == 0:
                unique_chunks.append(chunk)
                content_hashes[chunk_hash] += 1
        return unique_chunks
    
    unique_chunks = deduplicate_chunks(all_chunks)
    print(f"Total unique chunks after deduplication: {len(unique_chunks)}")

    # Initialize FAISS
     
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Try increasing to 5 matches
    print("FAISS database ready!")

def get_context(query):
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n".join(f"## {doc.metadata.get('source', 'Unknown File')}\n{doc.page_content}\n" for doc in docs)

def query_together(query, context=""):
    """Sends query to Together.AI's Mixtral-8x7B model."""
    prompt = f"<s>[INST] Answer the question in a simple sentence based only on the following context:\n{context}\n\nQuestion: {query} [/INST]"
    response = client.completions.create(
        model="mistralai/Mixtral-8x7B-v0.1",
        prompt=prompt,
        temperature=0.7,
        max_tokens=128,
        top_p=0.9
    )
    return response.choices[0].text.strip()

# Interactive query loop
while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break
    context = get_context(query)
    response = query_together(query, context)
    print("\n--- Response ---\n", response)

print("Script Complete")
