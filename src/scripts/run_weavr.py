import sys
import os
import argparse
import json
import re  # Add missing import for regex operations
import traceback  # For detailed error tracking
import shutil
import threading

# ✅ Ensure Python finds 'modules/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from modules.config_loader import load_api_key, get_model_name, set_model_name, get_model_api_name, load_config
from modules.retrieval import get_context
from modules.generation import query_together
from modules.file_change_handler import start_observer, stop_observer

# Enable debugging - commented out for production
DEBUG_MODE = False  # Change to True for debugging

def debug_print(message):
    """Print debug information when debug mode is enabled"""
    if DEBUG_MODE:
        print(f"🔍 DEBUG: {message}")

# ✅ Argument Parser for CLI
parser = argparse.ArgumentParser(description="Weavr AI - Interactive AI Assistant with Model & RAG Switching")
parser.add_argument("--rag", action="store_true", help="Enable Retrieval-Augmented Generation (RAG)")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

# Set debug mode from arguments
if args.debug:
    DEBUG_MODE = True

# ✅ Check if running interactively or in TUI mode
IS_TUI_MODE = not sys.stdin.isatty()

# ✅ Set Retrieval Mode
USE_RAG = args.rag
retriever = None
observer = None
reindex_flag = threading.Event()

if USE_RAG:
    if IS_TUI_MODE:
        print(json.dumps({"type": "status", "message": "Waiting for knowledge base path..."}))
        knowledge_base_dir = sys.stdin.readline().strip()
    else:
        knowledge_base_dir = input("📂 Enter the path to your knowledge directory (or press Enter to skip): ").strip()

    if knowledge_base_dir and os.path.exists(knowledge_base_dir) and os.path.isdir(knowledge_base_dir):
        print(f"✅ Using knowledge directory: {knowledge_base_dir}")

        # Initialize the retriever with persistent indexing
        from modules.persistent_indexing import get_or_create_persistent_index
        from modules.retrieval import get_faiss_retriever

        try:
            # Get or create persistent index
            vectorstore = get_or_create_persistent_index(knowledge_base_dir)

            if vectorstore is None:
                print("❌ FAISS failed to load. Retrieval will not work.")
                USE_RAG = False
            else:
                print("✅ FAISS Index Loaded Successfully!")
                retriever = get_faiss_retriever(vectorstore)
                observer = start_observer(knowledge_base_dir, reindex_flag)
        except Exception as e:
            print(f"❌ Error initializing RAG: {str(e)}")
            traceback.print_exc()
            USE_RAG = False

    else:
        print("⚠️ No valid directory provided. Running AI without document retrieval.")
        USE_RAG = False

if not IS_TUI_MODE:
    # ✅ Show Startup Info (CLI Mode Only)
    print(f"\n🕷️ Weavr AI is running with {'knowledge retrieval ✅' if USE_RAG else 'no retrieval ❌'}")
    print(f"🔹 Current AI Model: {get_model_name()} ({get_model_api_name()})")

    # ✅ Command Menu
    print("\n🔹 Commands:")
    print(" - `/exit` : Quit the program")
    print(" - `/model` : Switch AI models (by number)")
    print(" - `/rag` : Toggle Retrieval-Augmented Generation (RAG)")
    print(" - `/cot` : Toggle Chain-of-Thought reasoning")
    print(" - `/clearindex` : Clear the existing index to force re-indexing")
    print(" - Any other text: Send a query to AI\n")

# ✅ Main Loop for AI Interaction
conversation_history = ""
USE_COT = False  # ✅ New flag for CoT toggle
context = "No relevant retrieval data available."  # Initialize context globally

while True:
    try:
        if IS_TUI_MODE:
            query = sys.stdin.readline().strip()
            if not query:
                continue
        else:
            print("\n🎭 --- User --- 🎭")
            query = input("> ").strip()

        if query.lower() == "/exit":
            if IS_TUI_MODE:
                print(json.dumps({"type": "exit"}))
                sys.stdout.flush()
            print("Exiting Weavr AI...")
            break

        elif query.lower() == "/clearindex":
            # Clear the existing index
            from modules.persistent_indexing import PersistentIndexManager
            
            # Initialize the index manager
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

        elif query.lower() == "/model":
            # ✅ Show numbered list of available models
            config = load_config()
            models = list(config.get("together_ai", {}).get("models", {}).items())

            current_index = next((i for i, (key, _) in enumerate(models) if key == get_model_name()), -1)
            new_index = (current_index + 1) % len(models)
            new_model_key = models[new_index][0]
            set_model_name(new_model_key)

            if IS_TUI_MODE:
                print(json.dumps({"type": "model", "new_model": models[new_index][1]}))
                sys.stdout.flush()
            else:
                print(f"✅ Model switched to {models[new_index][1]} ({new_model_key})")

        elif query.lower() == "/rag":
            # Toggle Retrieval Mode
            USE_RAG = not USE_RAG
            status = "ENABLED" if USE_RAG else "DISABLED"

            if USE_RAG:
                print("\n📂 Enter the path to your knowledge directory:")
                knowledge_base_dir = input("> ").strip()

                if not os.path.exists(knowledge_base_dir) or not os.path.isdir(knowledge_base_dir):
                    print("❌ Invalid directory! RAG will remain disabled.")
                    USE_RAG = False
                else:
                    print(f"✅ Using knowledge directory, please wait...")

                    # Initialize the retriever with persistent indexing
                    from modules.persistent_indexing import get_or_create_persistent_index
                    from modules.retrieval import get_faiss_retriever

                    try:
                        # Get or create persistent index
                        vectorstore = get_or_create_persistent_index(knowledge_base_dir)
                        
                        if vectorstore is None:
                            print("❌ FAISS failed to load. Retrieval will not work.")
                            USE_RAG = False
                        else:
                            print("✅ FAISS Index Loaded Successfully!")
                            retriever = get_faiss_retriever(vectorstore)
                            observer = start_observer(knowledge_base_dir, reindex_flag)
                    except Exception as e:
                        print(f"❌ Error initializing RAG: {str(e)}")
                        traceback.print_exc()
                        USE_RAG = False

            if IS_TUI_MODE:
                print(json.dumps({"type": "rag", "status": status}))
                sys.stdout.flush()
            else:
                print(f"🔄 RAG Mode {status}")

        elif query.lower().startswith("/cot"):
            parts = query.split()
            if len(parts) == 1:
                # Toggle between disabled and default CoT
                USE_COT = not USE_COT
                cot_mode = "default"
                status = "ENABLED" if USE_COT else "DISABLED"
                print(f"🔄 Chain-of-Thought (CoT) Reasoning {status} (Mode: {cot_mode})")
            else:
                cot_mode = parts[1]
                USE_COT = True
                print(f"🔄 Chain-of-Thought (CoT) Reasoning ENABLED (Mode: {cot_mode})")

        else:
            # Check if reindexing is needed
            if reindex_flag.is_set():
                print("🔄 Reindexing due to file changes...")
                from modules.persistent_indexing import get_or_create_persistent_index
                from modules.retrieval import get_faiss_retriever

                try:
                    vectorstore = get_or_create_persistent_index(knowledge_base_dir, force_rebuild=True)
                    if vectorstore is None:
                        print("❌ FAISS failed to load. Retrieval will not work.")
                        context = "No relevant retrieval data available."
                    else:
                        retriever = get_faiss_retriever(vectorstore)
                        context = get_context(query, retriever)  # Get context after re-initialization
                        reindex_flag.clear()  # Clear the flag after reindexing
                except Exception as e:
                    print(f"❌ Error reindexing: {str(e)}")
                    traceback.print_exc()
                    context = "No relevant retrieval data available."
            else:
                # Retrieve context if RAG is enabled
                if USE_RAG and retriever:
                    context = get_context(query, retriever)
                    if "❌ Retrieval Failed" in context or not context.strip():
                        print("❌ FAISS retrieval failed. AI is generating a response without knowledge base data.")
                        context = "No relevant retrieval data available."
                    else:
                        print("✅ Retrieved Context:\n", context)
                else:
                    context = "No relevant retrieval data available"

            try:
                # Generate response with or without CoT
                response, token_count, reasoning_steps = query_together(
                    query, 
                    context, 
                    task_type="cot" if USE_COT else "default"
                )
                
                # Print CoT reasoning steps if available
                if USE_COT and reasoning_steps:
                    print("\n🔍 Chain-of-Thought Reasoning:")
                    for i, step in enumerate(reasoning_steps):
                        # Strip 'Step X:' prefix and clean up formatting for display
                        step_text = re.sub(r'^Step \d+:\s*', '', step)
                        print(f"   • {step_text.strip()}")
                
                # Print AI Response - completely separate from the reasoning steps
                print("\n🕷️ --- Weavr AI --- 🕷️")
                
                # Format the answer as a coherent narrative without the steps
                if USE_COT and "=== ANSWER ===" in response:
                    parts = response.split("=== ANSWER ===")
                    answer_part = parts[1].strip() if len(parts) > 1 else "No final answer provided."
                    
                    # Just print the final answer without the steps
                    print(answer_part)
                else:
                    # Regular non-CoT response
                    print(response.strip())
                
                print(f"\n🪙 Tokens Used: {token_count}")

            except Exception as query_error:
                print(f"❌ Error: {str(query_error)}")
                continue

    except EOFError:
        print("❌ TUI connection closed. Exiting Weavr AI...")
        break
    except Exception as e:
        if IS_TUI_MODE:
            print(json.dumps({"type": "error", "message": str(e)}))
            sys.stdout.flush()
        else:
            print(f"❌ Error: {str(e)}")
            print("💡 For more details, try running with the --debug flag")

print("Script Complete")

# Stop the observer if running
if observer:
    stop_observer(observer)
