import sys
import os

# Force working directory to src/
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure Python can find `modules/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.config_loader import load_api_key
from modules.retrieval import get_context
from modules.generation import query_together

# Ensure FAISS and other parts are ready before querying
USE_RAG = input("Would you like to work from a specific knowledge directory? (yes/no): ").strip().lower() == 'yes'

# Only load the retriever if we're using RAG
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

# Initialize conversation history
conversation_history = ""

# Show only relevant startup logs
if USE_RAG:
    print(f"\n🚀 Weavr AI is running with a knowledge directory ✅")
    print(f"📂 Knowledge Directory: {knowledge_base_dir}")
else:
    print(f"\n🚀 Weavr AI is running with no retrieval ❌")

while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    # Retrieve context if RAG is enabled
    context = get_context(query, retriever) if USE_RAG else ""

    # Get AI response and token count
    response, token_count = query_together(query, context)

    # Append AI response to the conversation history
    conversation_history += f"User: {query}\nAI: {response}\n"

    # Display response and token count
    print("\n--- Response ---\n", response)
    print(f"🔹 Tokens used: {token_count}")  # ✅ Keep this, but remove cluttered logs

print("Script Complete")
