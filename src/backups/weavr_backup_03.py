import requests
from langchain_chroma import Chroma  # Correct import for Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings  # Correct import for HuggingFaceEmbeddings
from config_loader import load_api_key  # Import the config loader to load the API key

# Load the Mistral API key securely using config_loader
MISTRAL_API_KEY = load_api_key()  # This will load the API key from config.yaml

# Set up headers for authentication with Mistral
headers = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

# Initialize the embedding model (using HuggingFace's embeddings from langchain_community)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Chroma DB for document retrieval (with the embedding function)
persist_directory = "db"
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# Set up the retriever with the correct method for getting relevant documents
retriever = db.as_retriever(search_kwargs={"k": 3})  # Correct method for getting relevant docs

# Create prompt template
template = """Answer the question based on the following context:
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Function to query Mistral API (Mixtral 8x7B)
def send_query_to_mistral(query, context):
    url = "https://api.mistral.ai/v1/models/mistral-8x7b-instruct-v0.1/query"  # Correct endpoint for Mistral
    
    # Construct the request payload
    payload = {
        "model": "mistral-8x7b-instruct-v0.1",  # Specify the Mixtral 8x7B model
        "messages": [{"role": "user", "content": query}],  # Use query as user message
        "context": context,  # Using context from Chroma DB
        "temperature": 0.7
    }
    
    # Send the request to Mistral API
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()  # Parse and return the response
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Query loop
while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    try:
        # Retrieve context from ChromaDB
        context = retriever.invoke(query)  # Use invoke() here instead of get_relevant_documents()
        
        # Send the query to Mistral and get the response
        response = send_query_to_mistral(query, context)

        if response:
            print(f"Response from Mixtral 8x7B: {response['choices'][0]['message']['content']}")
        else:
            print("No valid response received from the API.")
    
    except Exception as e:
        print(f"Error: {e}")
