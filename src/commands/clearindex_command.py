import shutil
import os
import traceback
from modules.persistent_indexing import PersistentIndexManager
from modules.file_change_handler import stop_observer

def execute(knowledge_base_dir, observer):
    """Executes the /clearindex command to clear the existing index."""
    # Clear the existing index
    manager = PersistentIndexManager()
    
    # Get the index directory
    index_dir = manager.index_dir
    
    try:
        # Remove the index directory
        shutil.rmtree(index_dir)
        
        # Recreate the index directory
        os.makedirs(index_dir, exist_ok=True)
        
        # Clear the index manifest
        manager.manifest["indices"] = {}
        manager._save_manifest()
        
        print(f"✅ Successfully cleared the index at {index_dir}")
        
        # Reset knowledge_base_dir and retriever, disable RAG
        knowledge_base_dir = None
        retriever = None
        USE_RAG = False
        print("🔄 RAG has been disabled. Please re-enable RAG to re-index the knowledge base.")
        
        # Stop the observer if running
        if observer:
            stop_observer(observer)
            observer = None
        
    except Exception as e:
        print(f"❌ Error clearing index: {str(e)}")
        traceback.print_exc()

    return knowledge_base_dir, None, observer, False
