import os
import traceback
from modules.file_change_handler import start_observer
from modules.persistent_indexing import get_or_create_persistent_index
from modules.retrieval import get_faiss_retriever
import sys
import threading

def initialize_rag(is_tui_mode, reindex_flag):
    """Initializes RAG components and returns the knowledge base directory, retriever, and observer."""
    knowledge_base_dir = None
    retriever = None
    observer = None

    if is_tui_mode:
        print("Waiting for knowledge base path...")
        knowledge_base_dir = sys.stdin.readline().strip()
    else:
        knowledge_base_dir = input("📂 Enter the path to your knowledge directory (or press Enter to skip): ").strip()

    if knowledge_base_dir and os.path.exists(knowledge_base_dir) and os.path.isdir(knowledge_base_dir):
        print(f"✅ Using knowledge directory: {knowledge_base_dir}")

        # Initialize the retriever with persistent indexing
        try:
            # Get or create persistent index
            vectorstore = get_or_create_persistent_index(knowledge_base_dir)

            if vectorstore is None:
                print("❌ FAISS failed to load. Retrieval will not work.")
            else:
                print("✅ FAISS Index Loaded Successfully!")
                retriever = get_faiss_retriever(vectorstore)
                observer = start_observer(knowledge_base_dir, reindex_flag)
        except Exception as e:
            print(f"❌ Error initializing RAG: {str(e)}")
            traceback.print_exc()
    else:
        print("⚠️ No valid directory provided. Running AI without document retrieval.")

    return knowledge_base_dir, retriever, observer
