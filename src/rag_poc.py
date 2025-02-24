import os
import markdown
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import hashlib
from collections import defaultdict

def parse_markdown_file(filepath):  # Correct indentation here
    with open(filepath, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    html = markdown.markdown(markdown_content)
    text = ''.join(BeautifulSoup(html, "html.parser").find_all(string=True))
    return text

obsidian_vault_path = os.path.join(os.path.expanduser("~"), "Documents", "The Scriptorium")

if not os.path.exists(obsidian_vault_path):
    raise ValueError(f"Obsidian vault not found at: {obsidian_vault_path}")

weavr_knowledge_base_path = os.path.join(obsidian_vault_path, "weavr_knowledge_base")

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

if documents:
    # 1. Splitting
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
        return_each_line=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=["\n## ", "\n# ", "\n\n", "\n", " "],
    )

    all_chunks = []
    for doc in documents:
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        final_chunks = text_splitter.split_documents(md_header_splits)
        all_chunks.extend(final_chunks)

    print(f"Total chunks after splitting: {len(all_chunks)}")

    # 2. Deduplication
    content_hashes = defaultdict(int)

    def deduplicate_chunks(chunks):
        unique_chunks = []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
            if content_hashes[chunk_hash] == 0:
                unique_chunks.append(chunk)
                content_hashes[chunk_hash] += 1
        return unique_chunks

    unique_chunks = deduplicate_chunks(all_chunks)

    print(f"Total unique chunks after deduplication: {len(unique_chunks)}")

    # 3. Embedding (Corrected)
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")  # Use updated class

    # 4. Chroma Database
    persist_directory = "db"
    db = Chroma.from_documents(unique_chunks, embeddings, persist_directory=persist_directory)

    print("Chroma database created successfully!")

print("Script Complete")