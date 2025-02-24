from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# After your existing db setup, replace the RetrievalQA setup with:
retriever = db.as_retriever(search_kwargs={"k": 3})

# Create prompt template
template = """Answer the question based on the following context:
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize pipeline using your handler
from transformers import pipeline
pipe = pipeline("text2text-generation", model="your_model_path", device=-1)  # -1 for CPU
llm = HuggingFacePipeline(pipeline=pipe)

# Create RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Replace your while loop with:
while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    try:
        response = rag_chain.invoke(query)
        print(response)
    except Exception as e:
        print(f"Error: {e}")