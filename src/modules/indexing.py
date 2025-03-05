import os
import markdown
from bs4 import BeautifulSoup
from pathlib import Path
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import traceback
import re

from langchain_together import TogetherEmbeddings

from modules.config_loader import load_api_key
from modules.parsers.file_parser import parse_file

def parse_markdown_file(filepath):
    """Parses a Markdown file and extracts text with structure preserved."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Extract headers and structure before converting to HTML
        headers = extract_markdown_headers(markdown_content)
        
        html = markdown.markdown(markdown_content)
        text = ''.join(BeautifulSoup(html, "html.parser").find_all(string=True))
        
        return text, headers
    except Exception as e:
        print(f"⚠️ Error parsing {filepath}: {e}")
        return "", []

def extract_markdown_headers(markdown_content):
    """Extract headers and their content from markdown text."""
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    headers = []
    
    for match in header_pattern.finditer(markdown_content):
        level = len(match.group(1))
        text = match.group(2)
        position = match.start()
        headers.append({
            "level": level,
            "text": text,
            "position": position
        })
    
    return headers

def split_with_metadata(text, filepath, headers=None):
    """Split text into chunks while preserving metadata about structure."""
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Basic metadata
    base_metadata = {
        "source": filepath,
        "filename": os.path.basename(filepath),
        "directory": os.path.dirname(filepath),
    }
    
    # Add file depth in directory structure
    relative_path = os.path.normpath(filepath)
    path_parts = relative_path.split(os.sep)
    base_metadata["path_depth"] = len(path_parts)
    
    # Find closest header for each chunk
    chunks = text_splitter.split_text(text)
    docs = []
    
    for i, chunk in enumerate(chunks):
        # Clone the base metadata
        metadata = dict(base_metadata)
        metadata["chunk_index"] = i
        
        # If we have headers, find the closest header to this chunk
        if headers:
            chunk_start_pos = text.find(chunk)
            if chunk_start_pos >= 0:
                # Find the closest header before this chunk
                closest_header = None
                for header in headers:
                    if header["position"] < chunk_start_pos:
                        if closest_header is None or header["position"] > closest_header["position"]:
                            closest_header = header
                
                if closest_header:
                    metadata["closest_header"] = closest_header["text"]
                    metadata["header_level"] = closest_header["level"]
        
        # Add document to list
        doc = {"content": chunk, "metadata": metadata}
        docs.append(doc)
    
    return docs

def get_modified_files(knowledge_base_dir, last_index_time):
    """Returns a list of modified files since last index time."""
    modified_files = []
    for root, _, files in os.walk(knowledge_base_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                if os.path.getmtime(full_path) > last_index_time:
                    modified_files.append(full_path)
    return modified_files

def load_document(file_path: str) -> List[dict]:
    """Load a single document with appropriate parser and preserve structure."""
    try:
        print(f"Loading document: {file_path}")
        
        # Special handling for markdown files
        if file_path.endswith('.md'):
            text_content, headers = parse_markdown_file(file_path)
            if not text_content:
                return []
                
            return split_with_metadata(text_content, file_path, headers)
        
        # Parse other files based on their type
        parsed_data = parse_file(file_path)
        
        if 'error' in parsed_data and parsed_data['error']:
            print(f"Error parsing {file_path}: {parsed_data['error']}")
            return []
        
        # Handle Python files specially
        if parsed_data['type'] == 'python_file':
            chunks = []
            
            # Add the summary as a chunk
            chunks.append({
                "content": parsed_data["summary"],
                "metadata": {
                    "source": file_path,
                    "type": "python_summary"
                }
            })
            
            # Add function and class documentation as chunks
            for func_name, func_details in parsed_data.get("functions", {}).items():
                docstring = parsed_data.get("docstrings", {}).get(f"function:{func_name}")
                function_content = f"Function: {func_name}\n"
                function_content += f"Arguments: {', '.join(func_details.get('arguments', []))}\n"
                if docstring:
                    function_content += f"Docstring: {docstring}"
                
                chunks.append({
                    "content": function_content,
                    "metadata": {
                        "source": file_path,
                        "type": "python_function",
                        "name": func_name
                    }
                })
            
            for class_name, class_details in parsed_data.get("classes", {}).items():
                class_content = f"Class: {class_name}\n"
                if class_details.get("bases"):
                    class_content += f"Inherits from: {', '.join(class_details['bases'])}\n"
                
                docstring = parsed_data.get("docstrings", {}).get(f"class:{class_name}")
                if docstring:
                    class_content += f"Docstring: {docstring}\n"
                
                if class_details.get("methods"):
                    class_content += f"Methods: {', '.join(class_details['methods'])}"
                
                chunks.append({
                    "content": class_content,
                    "metadata": {
                        "source": file_path,
                        "type": "python_class",
                        "name": class_name
                    }
                })
                
                # Add method docs
                for method in class_details.get("methods", []):
                    method_docstring = parsed_data.get("docstrings", {}).get(f"method:{class_name}.{method}")
                    if method_docstring:
                        chunks.append({
                            "content": f"Method: {class_name}.{method}\nDocstring: {method_docstring}",
                            "metadata": {
                                "source": file_path,
                                "type": "python_method",
                                "name": f"{class_name}.{method}"
                            }
                        })
            
            return chunks
        else:
            # For other file types, use text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            chunks = text_splitter.create_documents(
                [parsed_data.get("content", "")], 
                metadatas=[{"source": file_path}]
            )
            
            return [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
    except Exception as e:
        print(f"Error loading document {file_path}: {str(e)}")
        traceback.print_exc()
        return []

def load_documents(directory_path: str) -> List[dict]:
    """Load all documents from a directory recursively"""
    documents = []
    
    # Skip directories and file patterns
    skip_dirs = ['venv', '__pycache__', '.git', 'node_modules', '.pytest_cache']
    skip_extensions = ['.pyc', '.pyd', '.dll', '.so', '.dylib', '.exe', '.bin', '.dat', '.pkl', '.h5', '.pth']
    
    print(f"Scanning directory: {directory_path}")
    
    for root, dirs, files in os.walk(directory_path):
        # Skip directories in skip_dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        # Process each file
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
                
            # Skip files with extensions in skip_extensions
            if any(file.endswith(ext) for ext in skip_extensions):
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                doc_chunks = load_document(file_path)
                documents.extend(doc_chunks)
                if doc_chunks:
                    print(f"✓ Added {len(doc_chunks)} chunks from {file_path}")
                else:
                    print(f"✗ No chunks added from {file_path}")
            except Exception as e:
                print(f"✗ Error processing {file_path}: {str(e)}")
    
    print(f"📄 Total document chunks loaded: {len(documents)}")
    return documents

def load_or_create_faiss(documents: List[dict], embedding_model):
    """Load or create FAISS index from documents"""
    try:
        if not documents:
            print("No documents provided for indexing.")
            return None
        
        print(f"Creating FAISS index from {len(documents)} document chunks...")
        
        # Convert document format to what FAISS expects
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
            )
        
        # Create FAISS index
        vectorstore = FAISS.from_documents(langchain_docs, embedding_model)
        
        print("FAISS index created successfully.")
        return vectorstore
        
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")
        traceback.print_exc()
        return None
