import os
import requests
from langchain.prompts import ChatPromptTemplate

# Together.AI Configuration
from config_loader import load_api_key
from together import Together

# 🔹 Ask the user whether to use RAG BEFORE anything else runs
use_rag = input("Would you like to enable RAG (Retrieval-Augmented Generation)? (yes/no): ").strip().lower()
USE_RAG = use_rag in ["yes", "y"]

# Load API Key
TOGETHER_API_KEY = load_api_key()
client = Together(api_key=TOGETHER_API_KEY)

# Import FAISS-based retrieval *AFTER* checking the RAG setting
if USE_RAG:
    from rag_poc import get_context  # Import only if RAG is enabled

chat_history = []  # 🔹 Stores conversation history

def query_together(query):
    """Retrieves relevant context (if RAG is enabled) and queries the AI model."""
    global chat_history

    # 🔹 Adjust context handling
    context = get_context(query) if USE_RAG else ""

    # 🔹 If no context is found, avoid restrictive language
    if not context.strip():
        context = "Use your imagination and provide a complete response."

    # 🔹 Store query in chat history for better follow-up handling
    chat_history.append(f"User: {query}")
    if len(chat_history) > 5:  # Keep only last 5 exchanges
        chat_history = chat_history[-5:]

    # 🔹 Fixed prompt formatting
    full_prompt = f"""You are an AI assistant. Answer the user's request fully and thoughtfully.

### Context:
{context}

### Chat History:
{" ".join(chat_history)}

User: {query}
AI:
"""

    try:
        response = client.completions.create(
            model="mistralai/Mixtral-8x7B-v0.1",
            prompt=full_prompt,
            temperature=0.8,  # 🔹 Increased for more creativity
            max_tokens=300,   # 🔹 Allow longer responses
            top_p=0.9
        )
        
        answer = response.choices[0].text.strip()
        chat_history.append(f"AI: {answer}")  # 🔹 Store AI response in history
        return answer
    except Exception as e:
        return f"API Error: {e}"

# Interactive Query Loop
print(f"\n🚀 Weavr AI is running with {'RAG enabled ✅' if USE_RAG else 'RAG disabled ❌'}\n")
while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    response = query_together(query)
    print("\n--- Response ---\n", response)

print("Script Complete")

