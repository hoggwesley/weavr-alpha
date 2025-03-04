from langchain_community.vectorstores import FAISS

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

def get_context(query, retriever, top_k=10):
    """
    Retrieves relevant document chunks for a given query.
    Uses retriever.get_relevant_documents(query) from LangChain, which returns a list of Documents.
    Then applies sentence-level filtering to highlight query-related sentences.
    """
    docs = retriever.invoke(query) if retriever else []

    # If no results, return a default message instead of "retrieval failed"
    if not docs:
        return "No strong matches found in the knowledge base, but here's a general response."

    # **If one document is a near-perfect match, prioritize only that one**
    if len(docs) > 0 and query.lower() in docs[0].metadata.get("source", "").lower():
        docs = [docs[0]]  # Keep only the best match

    if not docs:
        return "No relevant documents found."

    extracted_texts = []
    for doc in docs:
        text = doc.page_content.strip()
        sentences = text.split(". ")

        # Prioritize sentences containing the query
        relevant_sentences = [s for s in sentences if query.lower() in s.lower()]

        # If too few, expand around those sentences
        if len(relevant_sentences) < 10:  # Increase threshold for better context
            idx = [i for i, s in enumerate(sentences) if query.lower() in s.lower()]
            for i in idx:
                start = max(0, i - 5)  # Pull 5 sentences before
                end = min(len(sentences), i + 6)  # Pull 6 sentences after
                relevant_sentences.extend(sentences[start:end])

        # Remove duplicates
        relevant_sentences = list(dict.fromkeys(relevant_sentences))

        extracted_text = ". ".join(relevant_sentences) + "..."
        if not extracted_texts:  # Add the document header only once
            extracted_texts.append(f"## {doc.metadata.get('source', 'Unknown File')}")

            extracted_texts.append(text)  # Append content without repeating the header

    return "\n".join(extracted_texts)

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
