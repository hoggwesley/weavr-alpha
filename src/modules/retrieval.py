from langchain_community.vectorstores import FAISS
import os
import re
from modules.query_enhancement import enhance_query
from rank_bm25 import BM25Okapi
import numpy as np

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
    Get the most relevant context for a query using hybrid search and re-ranking.
    """
    try:
        if not retriever:
            return "❌ Retrieval Failed: No retriever available."

        # Enhance the query for better retrieval
        enhanced_query = enhance_query(query)
        print(f"Enhanced query: {enhanced_query}")

        # Detect query intent and type
        query_lower = query.lower()
        is_code_query = any(keyword in query_lower for keyword in ['function', 'class', 'method', 'code', 'implementation'])
        is_file_query = any(ext in query_lower for ext in ['.py', '.md', '.txt', '.json', '.yaml', '.yml'])
        is_directory_query = any(keyword in query_lower for keyword in ['directory', 'folder', 'structure', 'organization'])

        # Smart Retrieval Decisions
        if is_code_query:
            # Focus on code-related documents and functions
            search_kwargs = {"filter": {"type": ["python_function", "python_class", "python_summary"]}}
        elif is_file_query:
            # Focus on specific files
            search_kwargs = {"filter": {"type": ["text", "model_file"]}}
        elif is_directory_query:
            # Focus on directory structure
            search_kwargs = {"filter": {"type": ["text"]}}
        else:
            # Default: no specific filter
            search_kwargs = {}
        
        # Dynamic k value based on query length
        query_length = len(query.split())
        if query_length <= 10:
            k = 5  # Short queries: retrieve fewer documents
        elif query_length <= 25:
            k = 7  # Medium queries: retrieve a moderate number of documents
        else:
            k = 10  # Long queries: retrieve more documents
        
        # Get documents from retriever (semantic search)
        docs = retriever.invoke(enhanced_query, **search_kwargs, k=k)

        if not docs:
            return "No relevant information found."

        # Re-ranking: Hybrid search with BM25 for lexical matching
        corpus = [doc.page_content for doc in docs]
        bm25 = BM25Okapi([text.lower().split() for text in corpus])
        bm25_scores = bm25.get_scores(enhanced_query.lower().split())

        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        normalized_bm25 = [score / max_bm25 for score in bm25_scores]

        # Combine scores: 0.7 * semantic + 0.3 * BM25
        combined_scores = []
        for i, doc in enumerate(docs):
            semantic_score = 1.0 - (i / len(docs))  # Approximate semantic score based on position
            combined_score = (0.7 * semantic_score) + (0.3 * normalized_bm25[i])
            combined_scores.append((doc, combined_score))

        # Sort by combined score
        reranked_docs = [doc for doc, _ in sorted(combined_scores, key=lambda x: x[1], reverse=True)]

        # Aggressive filtering: only keep documents with BM25 score > 0.1
        filtered_docs = [doc for i, doc in enumerate(reranked_docs) if normalized_bm25[i] > 0.1]

        # Format context from documents with improved structure
        context_parts = []
        seen_content = set()  # For deduplication

        for doc in filtered_docs[:5]:  # Limit to top 5 results
            # Skip duplicates
            if doc.page_content in seen_content:
                continue
            seen_content.add(doc.page_content)

            # Format with improved metadata awareness
            source = doc.metadata.get("source", "Unknown")
            filename = doc.metadata.get("filename", os.path.basename(source))
            directory = doc.metadata.get("directory", os.path.dirname(source))
            header = doc.metadata.get("closest_header", "")

            # Create a well-structured context entry
            context_entry = f"## {source}"
            if header:
                context_entry += f"\n{header}:"

            context_entry += f"\n{doc.page_content}"
            context_parts.append(context_entry)

        return "\n\n".join(context_parts)
    except Exception as e:
        import traceback
        traceback.print_exc()
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

    documents = load_documents(knowledge_base_dir, verbose=True)
    api_key = load_api_key()
    embedding_model = TogetherEmbeddings(model="BAAI/bge-large-en-v1.5", api_key=api_key)

    index = load_or_create_faiss(documents, embedding_model, verbose=True)
    retriever = get_faiss_retriever(index)

    if not retriever:
        print("❌ ERROR: Could not create a valid retriever from FAISS index.")
        sys.exit(1)

    query = input("Test query: ").strip()
    context = get_context(query, retriever)
    print(f"✅ Retrieved Context:\n{context}")
