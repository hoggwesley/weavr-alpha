import sys
import os
import argparse

# ✅ Correct the path so 'modules/' is properly found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # ✅ Adjusted for 'scripts/' directory
sys.path.append(PROJECT_ROOT)  # ✅ Ensures Python can locate 'modules/'

from modules.config_loader import load_api_key, get_model_name, set_model_name, get_model_api_name, load_config
from modules.retrieval import get_context
from modules.generation import query_together

# ✅ Argument Parser for CLI
parser = argparse.ArgumentParser(description="Weavr AI - Interactive AI Assistant with Model Switching")
parser.add_argument("--rag", action="store_true", help="Enable Retrieval-Augmented Generation (RAG)")
args = parser.parse_args()

# ✅ Set Retrieval Mode
USE_RAG = args.rag
retriever = None

if USE_RAG:
    knowledge_base_dir = input("Enter the path to your knowledge directory (or press Enter to skip): ").strip()
    
    if knowledge_base_dir and os.path.exists(knowledge_base_dir) and os.path.isdir(knowledge_base_dir):
        print(f"📂 Using knowledge directory: {knowledge_base_dir}")
        
        # Initialize the retriever (FAISS)
        from modules.indexing import load_documents, load_or_create_faiss
        from langchain_together import TogetherEmbeddings

        documents = load_documents(knowledge_base_dir)  # Load documents and chunk them
        embedding_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=load_api_key())
        retriever = load_or_create_faiss(documents, embedding_model).as_retriever(search_kwargs={"k": 5})
    else:
        print("⚠️ No valid directory provided. Running AI without document retrieval.")
        USE_RAG = False

# ✅ Show Startup Info
print(f"\n🚀 Weavr AI is running with {'knowledge retrieval ✅' if USE_RAG else 'no retrieval ❌'}")
print(f"🔹 Current AI Model: {get_model_name()} ({get_model_api_name()})")

# ✅ Command Menu
print("\n🔹 Commands:")
print(" - `/exit` : Quit the program")
print(" - `/model` : Switch AI models")
print(" - Any other text: Send a query to AI\n")

# ✅ Main Loop for AI Interaction
conversation_history = ""

while True:
    query = input("> ").strip()

    if query.lower() == "/exit":
        print("Exiting Weavr AI...")
        break

    elif query.lower() == "/model":
        # Show available models
        config = load_config()
        models = config.get("together_ai", {}).get("models", {})

        print("\nAvailable Models:")
        for model_key, model_name in models.items():
            print(f" - {model_key}: {model_name}")

        new_model = input("\nEnter the model key to switch to (e.g., 'qwen_72b_instruct'): ").strip().lower()

        # ✅ Allow case-insensitive matching
        valid_keys = {k.lower(): k for k in models.keys()}  # Create a case-insensitive lookup

        if new_model in valid_keys:
            set_model_name(valid_keys[new_model])  # ✅ Update config.yaml with the correct key
            print(f"✅ Model switched to {valid_keys[new_model]} ({models[valid_keys[new_model]]})")
        else:
            print(f"❌ Invalid model key! Please enter one of the following: {', '.join(models.keys())}")

        continue  # Restart loop after model change

    else:
        # Retrieve context if RAG is enabled
        context = get_context(query, retriever) if USE_RAG else ""

        # Get AI response and token count
        response, token_count = query_together(query, context)

        # Append AI response to the conversation history
        conversation_history += f"User: {query}\nAI: {response}\n"

        # Display response and token count
        print("\n--- AI Response ---\n", response)
        print(f"🔹 Tokens used: {token_count}")

print("Script Complete")

