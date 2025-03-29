import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Dosya yolunu doğru oluştur
PDF_STORAGE_DIR = r"C:\Users\mertm\Documents\pdfs"
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

def save_uploaded_file(uploaded_file):
    # Dosya yolunu düzgün bir şekilde oluştur
    file_path = os.path.join(PDF_STORAGE_DIR, uploaded_file.name)
    print("Kaydedilmeye çalışılan dosya yolu:", file_path)

    try:
        with open(file_path, "wb") as file:
            # Dosyayı okuma ve yazma işlemi
            file.write(uploaded_file.read())   
        return file_path
    except OSError as e:
        st.error(f"Dosya kaydedilemedi: {e}")
        return None

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# **Streamlit UI**
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

uploaded_pdf = st.file_uploader("Upload Research Document (PDF)", type="pdf")

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    
    if saved_path:
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        
        st.success("✅ Document processed successfully! Ask your questions below.")
        
        user_input = st.chat_input("Enter your question about the document...")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.spinner("Analyzing document..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)
                
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)
    else:
        st.error("Dosya kaydedilemedi. Lütfen tekrar deneyin.")
