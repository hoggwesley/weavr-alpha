import os
import markdown
from bs4 import BeautifulSoup
from pathlib import Path
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def parse_markdown_file(filepath):
    """Parses a Markdown file and extracts text."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        html = markdown.markdown(markdown_content)
        text = ''.join(BeautifulSoup(html, "html.parser").find_all(string=True))
        return text
    except Exception as e:
        print(f"⚠️ Error parsing {filepath}: {e}")
        return ""

def load_documents(knowledge_base_dir, verbose=False):
    """Loads and chunks Markdown files from the given directory."""
    documents = []
    processed_files = 0
    error_files = 0
    
    if verbose:
        print(f"📂 Checking for Markdown files in: {knowledge_base_dir}")

    if not os.path.exists(knowledge_base_dir):
        print("❌ ERROR: Knowledge base directory does not exist!")
        return []

    chunk_size = 500
    chunk_overlap = 50
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Get all files first, then process them
    all_files = []
    for root, _, files in os.walk(knowledge_base_dir):
        for file in files:
            if file.endswith(".md"):
                all_files.append(os.path.join(root, file))
    
    if verbose:
        print(f"📂 Found {len(all_files)} Markdown files")
    
    # Process each file
    for filepath in all_files:
        try:
            text_content = parse_markdown_file(filepath)
            if not text_content:
                continue
                
            if verbose and processed_files < 5:
                print(f"✅ Loaded: {os.path.basename(filepath)} ({len(text_content)} characters)")

            chunks = splitter.split_text(text_content)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk, 
                    metadata={"source": os.path.basename(filepath)}
                ))
            
            processed_files += 1
            
        except Exception as e:
            error_files += 1
            if verbose or error_files < 5:  # Show first few errors
                print(f"⚠️ Error processing {os.path.basename(filepath)}: {e}")

    if verbose:
        print(f"🔍 Processed {processed_files} files with {error_files} errors")
        print(f"🔍 Total chunks created: {len(documents)}")
    
    return documents

def load_or_create_faiss(chunks, embedding_model, verbose=False):
    """
    Creates or loads a FAISS index with proper LangChain compatibility.
    Returns a LangChain FAISS store that supports .as_retriever().
    """
    # Handle empty chunk scenario
    if not chunks:
        if verbose:
            print("⚠️ No document chunks provided. Returning an empty FAISS store.")
        return FAISS.from_texts(
            ["Placeholder document"], 
            embedding_model, 
            metadatas=[{"source": "Placeholder"}]
        )

    # Prepare texts and metadata
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # Generate embeddings
    embeddings = embedding_model.embed_documents(texts)
    
    # Create FAISS store using LangChain's API
    store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=embedding_model,
        metadatas=metadatas
    )
    
    # Save the index to a directory (not a file)
    index_dir = "data/faiss_index_store"
    
    # Make sure we're using a directory path, not a file path
    if os.path.exists(index_dir) and not os.path.isdir(index_dir):
        # If it exists but is a file, use a different path
        os.remove(index_dir)  # Remove the file
    
    # Create directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    try:
        store.save_local(index_dir)
        if verbose:
            print(f"✅ FAISS store saved to {index_dir}")
    except Exception as e:
        print(f"❌ ERROR saving FAISS store: {e}")
    
    return store
