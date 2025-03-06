from langchain_community.vectorstores import FAISS
import os
import re
from modules.query_enhancement import enhance_query
from rank_bm25 import BM25Okapi
import numpy as np
from modules.hybrid_search import HybridSearcher, combine_search_results

def get_faiss_retriever(index):
    """
    Wrap a LangChain FAISS store as a retriever, handling FlatL2 fallback.
    If the index is an instance of the FAISS vector store, we call .as_retriever().
    Otherwise, we return None (meaning it's a raw IndexFlatL2 or other).
    """
    if isinstance(index, FAISS):
        return index.as_retriever(search_kwargs={"k": 10})
    else:
        print("⚠️ FAISS index is raw (e.g., IndexFlatL2) and does not support .as_retriever().")
        return None

def get_context(query, retriever):
    """
    Retrieves context for a given query using a retriever.

    Args:
        query (str): The query to retrieve context for.
        retriever: The retriever object to use for retrieving context.

    Returns:
        str: The retrieved context, or an error message if retrieval fails.
    """
    try:
        context = retriever.invoke(query)
        context_str = "\n\n".join([doc.page_content for doc in context])
        return context_str
    except AttributeError as e:
        if "'InMemoryDocstore' object has no attribute 'values'" in str(e):
            return "❌ Retrieval Failed: The current index is not compatible with hybrid search. Please try a different index or disable hybrid search."
        else:
            return f"❌ Retrieval Failed: {str(e)}"
    except Exception as e:
        return f"❌ Retrieval Failed: {str(e)}"

if __name__ == "__main__":
    """
    This block is only for directly testing retrieval.py.
    You can run 'python retrieval.py' to do a quick test with hardcoded paths or data.
    """
    import sys
    from modules.config_loader import load_api_key
    from langchain_together import TogetherEmbeddings
    from modules.indexing import load_documents, load_or_create_faiss

    print("🔹 Running FAISS Retrieval Test...")

    # Hardcoded or prompt-based directory for quick test:
    knowledge_base_dir = input("Enter the path to your knowledge directory: ").strip()
    if not knowledge_base_dir or not os.path.isdir(knowledge_base_dir):
        print("❌ Invalid directory!")
        sys.exit(1)

    documents = load_documents(knowledge_base_dir)
    api_key = load_api_key()
    embedding_model = TogetherEmbeddings(model="BAAI/bge-large-en-v1.5", api_key=api_key)

    index = load_or_create_faiss(documents, embedding_model)
    retriever = get_faiss_retriever(index)

    if not retriever:
        print("❌ ERROR: Could not create a valid retriever from FAISS index.")
        sys.exit(1)

    query = input("Test query: ").strip()
    context = get_context(query, retriever)
    print(f"✅ Retrieved Context:\n{context}")
