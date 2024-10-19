import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama

# Function to load, split, and retrieve PDF documents
def load_and_retrieve_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Use Ollama embeddings with 'nomic-embed-text'
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Function to format the documents into a string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain for question answering
def rag_chain(pdf_path, question):
    retriever = load_and_retrieve_docs(pdf_path)
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Gradio interface for PDF-based question answering
iface = gr.Interface(
    fn=rag_chain,
    inputs=["file", "text"],  # Use 'file' input for PDF uploads and 'text' for the question
    outputs="text",
    title="RAG Chain Question Answering with PDFs",
    description="Upload a PDF and enter a query to get answers using the RAG chain."
)

# Launch the app
iface.launch()
