import sys
import os
import argparse
import json

# ✅ Ensure Python finds 'modules/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from modules.config_loader import load_api_key, get_model_name, set_model_name, get_model_api_name, load_config
from modules.retrieval import get_context
from modules.generation import query_together

# ✅ Argument Parser for CLI
parser = argparse.ArgumentParser(description="Weavr AI - Interactive AI Assistant with Model & RAG Switching")
parser.add_argument("--rag", action="store_true", help="Enable Retrieval-Augmented Generation (RAG)")
args = parser.parse_args()

# ✅ Check if running interactively or in TUI mode
IS_TUI_MODE = not sys.stdin.isatty()

# ✅ Set Retrieval Mode
USE_RAG = args.rag
retriever = None

if USE_RAG:
    if IS_TUI_MODE:
        print(json.dumps({"type": "status", "message": "Waiting for knowledge base path..."}))
        knowledge_base_dir = sys.stdin.readline().strip()
    else:
        knowledge_base_dir = input("📂 Enter the path to your knowledge directory (or press Enter to skip): ").strip()

    if knowledge_base_dir and os.path.exists(knowledge_base_dir) and os.path.isdir(knowledge_base_dir):
        print(f"✅ Using knowledge directory: {knowledge_base_dir}")

        # Initialize the retriever (FAISS)
        from modules.indexing import load_documents, load_or_create_faiss
        from langchain_together import TogetherEmbeddings

        documents = load_documents(knowledge_base_dir)  # Load documents and chunk them
        embedding_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=load_api_key())
        retriever = load_or_create_faiss(documents, embedding_model).as_retriever(search_kwargs={"k": 5})
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
    print(" - Any other text: Send a query to AI\n")

# ✅ Main Loop for AI Interaction
conversation_history = ""

while True:
    try:
        if IS_TUI_MODE:
            query = sys.stdin.readline().strip()
            if not query:  # Handle empty input in TUI mode
                continue
        else:
            print("\n🎭 --- User --- 🎭")
            query = input("> ").strip()

        if query.lower() == "/exit":
            if IS_TUI_MODE:
                print(json.dumps({"type": "exit"}))
                sys.stdout.flush()  # Ensure response is sent before exit
            print("Exiting Weavr AI...")
            break

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
            # ✅ Toggle Retrieval Mode
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

                    # ✅ Initialize RAG components
                    from modules.indexing import load_documents, load_or_create_faiss
                    from langchain_together import TogetherEmbeddings

                    documents = load_documents(knowledge_base_dir)
                    embedding_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=load_api_key())
                    retriever = load_or_create_faiss(documents, embedding_model).as_retriever(search_kwargs={"k": 5})

            if IS_TUI_MODE:
                print(json.dumps({"type": "rag", "status": status}))
                sys.stdout.flush()
            else:
                print(f"🔄 RAG Mode {status}")

        else:
            # Retrieve context if RAG is enabled
            context = get_context(query, retriever) if USE_RAG else ""

            # Get AI response and token count
            response, token_count = query_together(query, context)

            # Append AI response to the conversation history
            conversation_history += f"User: {query}\nAI: {response}\n"

            if IS_TUI_MODE:
                print(json.dumps({"type": "response", "text": response, "tokens": token_count}))
                sys.stdout.flush()
            else:
                print("\n🕷️ --- Weavr AI --- 🕷️")
                print(response)
                print(f"🪙 Tokens Used: {token_count}")

    except EOFError:
        print("❌ TUI connection closed. Exiting Weavr AI...")
        break
    except Exception as e:
        if IS_TUI_MODE:
            print(json.dumps({"type": "error", "message": str(e)}))
            sys.stdout.flush()
        else:
            print(f"❌ Error: {str(e)}")

print("Script Complete")
