"""
Module for persistent storage and incremental indexing of document embeddings.
"""
import os
import json
import time
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from langchain_community.vectorstores import FAISS
from modules.config_loader import load_api_key
from modules.indexing import load_document, load_or_create_faiss
from langchain_together import TogetherEmbeddings

# Constants
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "indices")
INDEX_RETENTION_DAYS = 30  # How long to keep indices
MANIFEST_FILENAME = "index_manifest.json"

class PersistentIndexManager:
    """
    Manages persistent storage and incremental updating of document indices.
    """
    def __init__(self, index_dir: Optional[str] = None, embedding_model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize the index manager.
        
        Args:
            index_dir: Directory to store indices. If None, uses default.
            embedding_model_name: Name of the embedding model to use.
        """
        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self.manifest_path = self.index_dir / MANIFEST_FILENAME
        self.manifest = self._load_manifest()
        self.embedding_model = None
        
    def _load_manifest(self) -> Dict:
        """Load the index manifest file or create if it doesn't exist."""
        if not self.manifest_path.exists():
            return {
                "version": 1,
                "embedding_model": self.embedding_model_name,
                "config_hash": "",
                "indices": {},
                "last_cleanup": datetime.now().isoformat()
            }
        
        with open(self.manifest_path, 'r') as f:
            return json.load(f)
    
    def _save_manifest(self) -> None:
        """Save the index manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _get_config_hash(self) -> str:
        """
        Generate a hash of the current embedding configuration.
        This helps detect when the model or parameters have changed.
        """
        # We can expand this to include more configuration parameters
        config_str = f"{self.embedding_model_name}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _index_path_for_directory(self, directory: str) -> Path:
        """Generate a unique path for a directory's index."""
        # Create a hash of the directory path to use as the index name
        dir_hash = hashlib.md5(directory.encode()).hexdigest()[:10]
        return self.index_dir / f"index_{dir_hash}"
    
    def _get_modified_files(self, directory: str, last_indexed_time: Dict[str, float]) -> List[str]:
        """
        Get files that have been modified since last indexed.
        
        Args:
            directory: Directory to scan
            last_indexed_time: Dictionary mapping file paths to their last indexed time
            
        Returns:
            List of modified file paths
        """
        modified_files = []
        
        # Skip directories and file patterns
        skip_dirs = ['venv', '__pycache__', '.git', 'node_modules', '.pytest_cache']
        skip_extensions = ['.pyc', '.pyd', '.dll', '.so', '.dylib', '.exe', '.bin', '.dat', '.pkl', '.h5', '.pth']
        
        for root, dirs, files in os.walk(directory):
            # Skip directories in skip_dirs
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.startswith('.') or any(file.endswith(ext) for ext in skip_extensions):
                    continue
                    
                file_path = os.path.join(root, file)
                mod_time = os.path.getmtime(file_path)
                
                # If file wasn't indexed before or has been modified
                if file_path not in last_indexed_time or mod_time > last_indexed_time[file_path]:
                    modified_files.append(file_path)
        
        return modified_files
    
    def _cleanup_old_indices(self) -> None:
        """Remove indices older than INDEX_RETENTION_DAYS."""
        try:
            # Parse the last cleanup time
            last_cleanup = datetime.fromisoformat(self.manifest["last_cleanup"])
            
            # Only run cleanup once per day
            if datetime.now() - last_cleanup < timedelta(days=1):
                return
                
            cutoff_date = datetime.now() - timedelta(days=INDEX_RETENTION_DAYS)
            
            # Get all directories in the index directory
            indices_to_remove = []
            
            for dir_entry in os.scandir(self.index_dir):
                if dir_entry.is_dir():
                    # Check if the directory is an index directory
                    if dir_entry.name.startswith("index_"):
                        # Check last modification time
                        dir_time = datetime.fromtimestamp(os.path.getmtime(dir_entry.path))
                        if dir_time < cutoff_date:
                            indices_to_remove.append(dir_entry.path)
            
            # Remove old indices
            for index_path in indices_to_remove:
                print(f"Removing old index: {os.path.basename(index_path)}")
                shutil.rmtree(index_path)
            
            # Update the manifest
            self.manifest["last_cleanup"] = datetime.now().isoformat()
            self._save_manifest()
            
        except Exception as e:
            print(f"Warning: Error during index cleanup: {str(e)}")
    
    def get_or_create_index(self, directory: str, force_rebuild: bool = False) -> Tuple[FAISS, List[str], int]:
        """
        Get an existing index or create a new one for the given directory.
        
        Args:
            directory: Directory to index
            force_rebuild: If True, rebuild the index even if not needed
            
        Returns:
            Tuple of (faiss_index, modified_files, document_count)
        """
        # Ensure embedding model is initialized
        if self.embedding_model is None:
            api_key = load_api_key()
            self.embedding_model = TogetherEmbeddings(
                model=self.embedding_model_name, 
                api_key=api_key
            )
        
        # Check if config has changed
        current_config_hash = self._get_config_hash()
        if current_config_hash != self.manifest.get("config_hash", ""):
            print(f"Embedding configuration has changed. Forcing rebuild of indices.")
            force_rebuild = True
            self.manifest["config_hash"] = current_config_hash
            self.manifest["embedding_model"] = self.embedding_model_name
        
        # Get the path for this directory's index
        index_path = self._index_path_for_directory(directory)
        
        # Get the last indexed state for this directory
        dir_index_info = self.manifest.get("indices", {}).get(directory, {
            "last_indexed": {},
            "document_count": 0,
            "last_update": "never"
        })
        
        # Get files that have been modified since last indexing
        last_indexed_time = {file: timestamp for file, timestamp in dir_index_info.get("last_indexed", {}).items()}
        modified_files = self._get_modified_files(directory, last_indexed_time)
        
        # Check if we need to rebuild the index
        index_exists = index_path.exists() and (index_path / "index.faiss").exists()
        
        if not index_exists or force_rebuild or modified_files:
            print(f"{'Creating' if not index_exists else 'Updating'} index for {directory}")
            
            # Load documents - either all or just modified
            documents = []
            
            if not index_exists or force_rebuild:
                # Load all documents in directory
                from modules.indexing import load_documents
                documents = load_documents(directory)
                
                # Update the last indexed time for all files
                new_last_indexed = {}
                for root, _, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        new_last_indexed[file_path] = os.path.getmtime(file_path)
                
                dir_index_info["last_indexed"] = new_last_indexed
                
            else:
                # Only load modified documents
                for file_path in modified_files:
                    try:
                        doc_chunks = load_document(file_path)
                        documents.extend(doc_chunks)
                        # Update the last indexed time
                        last_indexed_time[file_path] = os.path.getmtime(file_path)
                    except Exception as e:
                        print(f"Error loading document {file_path}: {str(e)}")
                
                dir_index_info["last_indexed"] = last_indexed_time
            
            # Store or update FAISS index
            if not index_exists or force_rebuild:
                # Create new index
                vectorstore = load_or_create_faiss(documents, self.embedding_model)
                if vectorstore:
                    os.makedirs(index_path, exist_ok=True)
                    try:
                        # Try with pickle_serialize parameter first
                        vectorstore.save_local(str(index_path), pickle_serialize=True)
                    except TypeError:
                        # Fall back to regular save if pickle_serialize is not supported
                        vectorstore.save_local(str(index_path))
            else:
                # Update existing index with new documents
                try:
                    # Load existing index with deserialization allowed
                    vectorstore = FAISS.load_local(
                        str(index_path), 
                        self.embedding_model, 
                        allow_dangerous_deserialization=True
                    )
                    
                    # If we have new documents, add them to the index
                    if documents:
                        # Convert to LangChain document format
                        from langchain.docstore.document import Document
                        langchain_docs = []
                        for doc in documents:
                            langchain_docs.append(
                                Document(
                                    page_content=doc["content"],
                                    metadata=doc["metadata"]
                                )
                            )
                        
                        # Add documents to the index
                        vectorstore.add_documents(langchain_docs)
                        
                        # Save the updated index
                        try:
                            vectorstore.save_local(str(index_path), pickle_serialize=True)
                        except TypeError:
                            vectorstore.save_local(str(index_path))
                    
                except Exception as e:
                    print(f"Error updating index: {str(e)}. Creating new index.")
                    # If update fails, create a new index
                    from modules.indexing import load_documents
                    all_documents = load_documents(directory)
                    vectorstore = load_or_create_faiss(all_documents, self.embedding_model)
                    if vectorstore:
                        os.makedirs(index_path, exist_ok=True)
                        try:
                            vectorstore.save_local(str(index_path), pickle_serialize=True)
                        except TypeError:
                            vectorstore.save_local(str(index_path))
            
            # Update index info in manifest
            dir_index_info["document_count"] = len(documents) if documents else dir_index_info["document_count"]
            dir_index_info["last_update"] = datetime.now().isoformat()
            self.manifest["indices"][directory] = dir_index_info
            self._save_manifest()
            
            return vectorstore, modified_files, dir_index_info["document_count"]
        
        else:
            # No changes, return existing index
            print(f"Using existing index for {directory} (last updated {dir_index_info['last_update']})")
            try:
                # Enable allow_dangerous_deserialization since we're loading our own files
                vectorstore = FAISS.load_local(
                    str(index_path), 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True
                )
                return vectorstore, [], dir_index_info["document_count"]
            except Exception as e:
                print(f"Error loading index: {str(e)}. Will rebuild.")
                # If loading fails, force a rebuild on next call
                return self.get_or_create_index(directory, force_rebuild=True)
    
    def cleanup(self):
        """Run cleanup tasks."""
        self._cleanup_old_indices()

def get_or_create_persistent_index(knowledge_dir: str, force_rebuild: bool = False) -> FAISS:
    """
    Get or create a persistent index for the given knowledge directory.
    This is a convenience function for use in other modules.
    
    Args:
        knowledge_dir: Directory containing knowledge files
        force_rebuild: Whether to force rebuilding the index
    
    Returns:
        FAISS index object
    """
    manager = PersistentIndexManager()
    vectorstore, modified_files, doc_count = manager.get_or_create_index(knowledge_dir, force_rebuild)
    
    if modified_files:
        print(f"Updated {len(modified_files)} files in the index.")
    print(f"Index contains {doc_count} document chunks.")
    
    # Run cleanup as a good practice
    manager.cleanup()
    
    return vectorstore
