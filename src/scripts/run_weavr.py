import sys
import os
import argparse
import json
import re  # Add missing import for regex operations
import traceback  # For detailed error tracking
import shutil
import threading
import subprocess

# ✅ Ensure Python finds 'modules/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from modules.config_loader import load_api_key, get_model_name, set_model_name, get_model_api_name, load_config
from modules.retrieval import get_context
from modules.generation import query_together
from modules.file_change_handler import start_observer, stop_observer
from modules.user_instructions import load_instructions, save_instructions
from commands import model_command, rag_command, cot_command, instructions_command, clearindex_command
from rag import initialize_rag

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
knowledge_base_dir = None

if USE_RAG:
    knowledge_base_dir, retriever, observer = initialize_rag(IS_TUI_MODE, reindex_flag)
    USE_RAG = retriever is not None

if not IS_TUI_MODE:
    # ✅ Show Startup Info (CLI Mode Only)
    print(f"\n🕷️ Weavr AI is running with {'knowledge retrieval ✅' if USE_RAG else 'no retrieval ❌'}")
    print(f"🔹 Current AI Model: {get_model_name()} ({get_model_api_name()})")

    # ✅ Command Menu
    print("\n🔹 Commands:")
    print(" - `/exit` : Quit the program")
    print(" - `/model` : Select AI model from menu")
    print(" - `/rag` : Toggle Retrieval-Augmented Generation (RAG)")
    print(" - `/cot` : Toggle Chain-of-Thought reasoning")
    print(" - `/clearindex` : Clear the existing index to force re-indexing")
    print(" - `/instructions` : View or edit AI behavior instructions")
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
            print("Exiting Weavr AI...")
            break

        elif query.lower() == "/clearindex":
            knowledge_base_dir, retriever, observer, USE_RAG = clearindex_command.execute(knowledge_base_dir, observer)
            
        elif query.lower() == "/model":
            try:
                model_command.execute()
                print(f"🔹 Current AI Model: {get_model_name()} ({get_model_api_name()})")  # Print the current model after switching
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                if DEBUG_MODE:
                    traceback.print_exc()

        elif query.lower() == "/rag":
            knowledge_base_dir, retriever, observer, USE_RAG = rag_command.execute(IS_TUI_MODE, knowledge_base_dir, reindex_flag, observer, USE_RAG)

        elif query.lower().startswith("/cot"):
            USE_COT = cot_command.execute(query, USE_COT)

        elif query.lower() == "/instructions":
            instructions_command.execute()

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
                    try:
                        context = get_context(query, retriever)
                        if "❌ Retrieval Failed" in context or not context.strip():
                            print("❌ FAISS retrieval failed. AI is generating a response without knowledge base data.")
                            context = "No relevant retrieval data available."
                        else:
                            # Comment out context retrieval output for cleaner UI
                            if DEBUG_MODE:
                                print("✅ Retrieved Context:\n", context)
                    except Exception as query_error:
                        print(f"❌ Error: {str(query_error)}")
                        context = "No relevant retrieval data available."
                else:
                    context = "No relevant retrieval data available."

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
                traceback.print_exc()
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
