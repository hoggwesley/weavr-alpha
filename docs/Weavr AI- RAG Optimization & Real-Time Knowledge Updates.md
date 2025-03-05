# **Weavr AI: RAG Optimization & Real-Time Knowledge Updates**

---

### **Objective:**

This update improves Weavr AI's Retrieval-Augmented Generation (RAG) system by:

- **Upgrading the embedding model** for better retrieval.
- **Expanding and tuning chunking** for better coherence.
- **Making retrieval smarter**, deciding when to return full documents, partial excerpts, or no retrieval at all.
- **Checking for new file updates before each query** and only indexing changes, not the full dataset.

---

## **1️⃣ Embedding Model Upgrade**

**Current Model:** `M2-BERT-Retrieval-8k` (cheap but weak at long-context retrieval) **New Model:** `BAAI-Bge-Large-1.5 ($0.02 per 1M tokens)`

🔹 **Why?**

- **Better long-document understanding**
- **More accurate similarity search**
- **Minimal cost increase (~2x current cost but significant quality boost)**

🔹 **Implementation in `indexing.py`**

```python
from langchain_community.embeddings import TogetherEmbeddings

embedding_model = TogetherEmbeddings(model="BAAI/Bge-Large-1.5", api_key=TOGETHER_API_KEY)
```

✅ **Effect:** ✔ Reduces **irrelevant chunk retrieval** ✔ Improves **semantic matching**

---

## **2️⃣ Smarter Chunking & Retrieval Expansion**

### **🔹 Dynamic Chunking**

Instead of **fixed-size chunks (~500 tokens)**, we will:

- **Increase base chunk size** to **1,500-2,000 tokens**
- **Dynamically adjust chunking based on query type**
    - **If retrieval confidence is low → Merge more chunks**
    - **If query is broad → Retrieve entire sections or files**

🔹 **Modify `load_documents()` in `indexing.py`**

```python
def load_documents(knowledge_base_dir):
    chunk_size = 2000  # Increased from 500
    chunk_overlap = 200  # Helps keep context
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
```

✅ **Effect:** ✔ Chunks retain **better context**, reducing randomness. ✔ **Larger, more meaningful sections** retrieved.

---

## **3️⃣ Real-Time File Change Detection**

### **🔹 Automatically Index New & Modified Files**

Instead of **re-indexing everything**, Weavr will **only update changed files** before a query.

🔹 **Modify `indexing.py` to check timestamps**

```python
import os

def get_modified_files(knowledge_base_dir, last_index_time):
    modified_files = []
    for root, _, files in os.walk(knowledge_base_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                if os.path.getmtime(full_path) > last_index_time:
                    modified_files.append(full_path)
    return modified_files
```

✅ **Effect:** ✔ Prevents unnecessary indexing. ✔ Keeps RAG updated in real-time without slowing down responses.

---

## **4️⃣ Smart Retrieval Decisions**

Weavr should **adapt retrieval based on query type**:

- If the query is **specific** → Use small chunks.
- If the query is **broad or vague** → Retrieve **entire sections or full files**.

🔹 **Modify `get_context()` in `retrieval.py`**

```python
def get_context(query, retriever):
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."
    
    if len(query.split()) < 5 or "summarize" in query.lower():
        return "\n\n".join([doc.page_content for doc in docs])  # Retrieve full section
    
    return "\n\n".join([doc.page_content for doc in docs[:3]])  # Top 3 chunks
```

✅ **Effect:** ✔ **Fewer cases where important context is missing.** ✔ More **intelligent balance between full-document vs. chunked retrieval**.

---

## **5️⃣ Hybrid FAISS + Keyword Search (Better Retrieval Precision)**

FAISS **sometimes retrieves "similar" but unimportant chunks**. We can fix this by **reranking the chunks intelligently**.

🔹 **Modify `retrieval.py` to include BM25**

```python
def hybrid_retrieval(query, retriever, bm25_search):
    faiss_docs = retriever.get_relevant_documents(query)
    bm25_docs = bm25_search(query)  # Uses BM25 for keyword matching
    merged_docs = list(set(faiss_docs + bm25_docs))  # Deduplicate and merge
    return merged_docs
```

✅ **Effect:** ✔ Fixes **cases where FAISS retrieves "similar but useless" text**. ✔ Ensures **keyword-heavy queries get direct document matches**.

---

## **🔹 Summary: What This Plan Fixes**

|**Problem**|**Fix**|**Effect**|
|---|---|---|
|**Chunks feel random**|🔹 **Better embedding model**|🔹 **More relevant retrieval**|
|**Context gets cut off**|🔹 **Larger base chunks**|🔹 **Preserves document flow**|
|**Some queries need whole sections**|🔹 **Smart retrieval expansion**|🔹 **Retrieves full files when needed**|
|**FAISS retrieves "similar but irrelevant" chunks**|🔹 **Hybrid FAISS + BM25 search**|🔹 **Fixes bad chunk selection**|

✅ **Minimal cost increase** but **massive accuracy improvement**. ✅ Keeps Weavr **API-based and easy to package** (no local models).

---

### **🚀 Next Steps: Implement & Test**

Do you want to:

1. **Upgrade the embedding model now** (`BAAI-Bge-Large-1.5`)?
2. **Expand chunking size immediately** (1,500-2,000 tokens)?
3. **Implement real-time indexing & retrieval optimization next?**

I can prepare **step-by-step implementation updates** for each. Let me know where you want to start! 🚀