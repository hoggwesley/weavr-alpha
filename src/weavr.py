import os

# 🔹 Ask the user whether to use RAG BEFORE anything else runs
use_rag = input("Would you like to enable RAG (Retrieval-Augmented Generation)? (yes/no): ").strip().lower()
USE_RAG = use_rag in ["yes", "y"]

# Import FAISS-based retrieval *AFTER* checking the RAG setting
if USE_RAG:
    from rag_poc import get_context  # Import only if RAG is enabled

chat_history = []  # 🔹 Stores conversation history

# Interactive Query Loop
print(f"\n🚀 Weavr AI is running with {'RAG enabled ✅' if USE_RAG else 'RAG disabled ❌'}\n")
while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    response = "Gemini Flash 2.0 is now the default model."
    print("\n--- Response ---\n", response)

print("Script Complete")

