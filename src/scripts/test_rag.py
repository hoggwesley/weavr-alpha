# test_rag.py
import sys
import os

# Ensure Python can find `modules/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.config_loader import load_api_key
from modules.indexing import load_documents, load_or_create_faiss
# 🔹 CRITICAL: Import both get_context and get_faiss_retriever
from modules.retrieval import get_context, get_faiss_retriever

from langchain_together import TogetherEmbeddings

# Ask for knowledge directory
knowledge_base_dir = input("Enter the path to your knowledge directory: ").strip()

if not os.path.exists(knowledge_base_dir):
    print("❌ ERROR: Invalid directory!")
    sys.exit(1)

print(f"📂 Using knowledge directory: {knowledge_base_dir}")

# 1. Load documents
documents = load_documents(knowledge_base_dir)
print(f"📂 Number of documents loaded: {len(documents)}")
for i, doc in enumerate(documents[:5]):
    print(f"📝 Doc {i+1}: {doc.metadata['source']} - {len(doc.page_content)} chars")

# 2. Load embeddings
TOGETHER_API_KEY = load_api_key()
embedding_model = TogetherEmbeddings(
    model="BAAI/bge-large-en-v1.5",
    api_key=TOGETHER_API_KEY
)

# 3. Build FAISS index
index = load_or_create_faiss(documents, embedding_model)

# 4. Wrap it in a retriever
retriever = get_faiss_retriever(index)
if not retriever:
    print("❌ ERROR: Could not create a valid retriever (possibly IndexFlatL2).")
    sys.exit(1)

# 5. Query loop
while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    context = get_context(query, retriever)
    print("\n--- Retrieved Context ---\n", context, "\n")
