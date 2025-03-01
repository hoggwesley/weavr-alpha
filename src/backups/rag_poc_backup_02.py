import os
import markdown
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_together import TogetherEmbeddings
import hashlib
from collections import defaultdict
from typing import List

# Together API Configuration
from config_loader import load_api_key
from together import Together

# Load API key from config.yaml
TOGETHER_API_KEY = load_api_key()
print(f"DEBUG: Using Together API Key = {TOGETHER_API_KEY[:6]}****** (Length: {len(TOGETHER_API_KEY)})")

# Ensure API key is not empty
if not TOGETHER_API_KEY:
    raise ValueError("ERROR: TOGETHER_API_KEY is missing from config.yaml!")

# Initialize Together Embeddings Model with API key explicitly passed
embedding_model = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    api_key=TOGETHER_API_KEY  # 🔹 Ensure it uses the correct API key
)

# Initialize Together LLM Client
client = Together(api_key=TOGETHER_API_KEY)

def parse_markdown_file(filepath):
    """Parses a Markdown file and extracts text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    html = markdown.markdown(markdown_content)
    text = ''.join(BeautifulSoup(html, "html.parser").find_all(string=True))
    return text

# Define Vault Path
obsidian_vault_path = os.path.join(os.path.expanduser("~"), "Documents", "The Scriptorium")
weavr_knowledge_base_path = os.path.join(obsidian_vault_path, "weavr_knowledge_base")

if not os.path.exists(weavr_knowledge_base_path):
    raise ValueError(f"Obsidian vault not found at: {weavr_knowledge_base_path}")

# Load documents
test_files = ["note1.md", "note2.md", "note3.md"]
documents = []

for filename in test_files:
    test_file_path = os.path.join(weavr_knowledge_base_path, filename)
    if os.path.exists(test_file_path):
        try:
            text_content = parse_markdown_file(test_file_path)
            print(f"{filename} parsed. Length: {len(text_content)}")
            documents.append(Document(page_content=text_content, metadata={"source": filename}))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"{filename} not found:", test_file_path)

print(f"Total documents loaded: {len(documents)}")

# Process documents if available
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    all_chunks = text_splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(all_chunks)}")

    # Deduplication
    content_hashes = defaultdict(int)

    def deduplicate_chunks(chunks):
        """Removes duplicate document chunks based on hash values."""
        unique_chunks = []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
            if content_hashes[chunk_hash] == 0:
                unique_chunks.append(chunk)
                content_hashes[chunk_hash] += 1
        return unique_chunks

    unique_chunks = deduplicate_chunks(all_chunks)
    print(f"Total unique chunks after deduplication: {len(unique_chunks)}")

    # FAISS Vector Store
    vectorstore = FAISS.from_documents(unique_chunks, embedding_model)
    retriever = vectorstore.as_retriever()
    print("FAISS database created successfully!")

def get_context(query):
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."
    
    # Extract filename if explicitly mentioned (e.g., "note1.md")
    target_file = None
    words = query.lower().split()
    for word in words:
        if word.endswith(".md"):
            target_file = word
            break

    filtered_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown File")

        # 🔹 If user specifies a file, filter to that file only
        if target_file and target_file != source:
            continue  # Skip mismatched files

        filtered_docs.append(f"## {source}\n{doc.page_content}\n")

    return "\n".join(filtered_docs) if filtered_docs else f"No relevant content found in {target_file}."

def query_together(query, context=""):
    """Sends query to Together.AI's Mixtral-8x7B model."""
    prompt = f"<s>[INST] Answer the question in a simple sentence based only on the following context:\n{context}\n\nQuestion: {query} [/INST]"
    response = client.completions.create(
        model="mistralai/Mixtral-8x7B-v0.1",
        prompt=prompt,
        temperature=0.7,
        max_tokens=128,
        top_p=0.9
    )
    return response.choices[0].text.strip()

# Interactive query loop
while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break
    context = get_context(query)
    response = query_together(query, context)
    print("\n--- Response ---\n", response)

print("Script Complete")
