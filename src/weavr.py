from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import pipeline

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the existing Chroma database
persist_directory = "db"
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Set up retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Create prompt template
template = """Answer the question based on the following context:
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize pipeline using your handler
pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)  # -1 for CPU
llm = HuggingFacePipeline(pipeline=pipe)

# Create RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query loop
while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    try:
        response = rag_chain.invoke(query)
        print(response)
    except Exception as e:
        print(f"Error: {e}")

