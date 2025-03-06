import os
import traceback
from modules.file_change_handler import start_observer, stop_observer
from modules.persistent_indexing import get_or_create_persistent_index
from modules.retrieval import get_faiss_retriever

def execute(is_tui_mode, knowledge_base_dir, reindex_flag, observer, use_rag):
    """Executes the /rag command to toggle Retrieval-Augmented Generation (RAG)."""
    use_rag = not use_rag
    status = "ENABLED" if use_rag else "DISABLED"

    if use_rag:
        print("\n📂 Enter the path to your knowledge directory:")
        knowledge_base_dir = input("> ").strip()

        if not os.path.exists(knowledge_base_dir) or not os.path.isdir(knowledge_base_dir):
            print("❌ Invalid directory! RAG will remain disabled.")
            use_rag = False
        else:
            print(f"✅ Using knowledge directory: {knowledge_base_dir}")

            # Initialize the retriever with persistent indexing
            try:
                # Get or create persistent index
                vectorstore = get_or_create_persistent_index(knowledge_base_dir)
                
                if vectorstore is None:
                    print("❌ FAISS failed to load. Retrieval will not work.")
                    use_rag = False
                else:
                    print("✅ FAISS Index Loaded Successfully!")
                    retriever = get_faiss_retriever(vectorstore)
                    observer = start_observer(knowledge_base_dir, reindex_flag)
            except Exception as e:
                print(f"❌ Error initializing RAG: {str(e)}")
                traceback.print_exc()
                use_rag = False
    else:
        if observer:
            stop_observer(observer)
            observer = None
        retriever = None
        knowledge_base_dir = None

    print(f"🔄 RAG Mode {status}")
    return knowledge_base_dir, retriever, observer, use_rag
