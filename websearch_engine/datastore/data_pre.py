import os
import re
from pathlib import Path
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# === CONFIG ===
VECTOR_DB_FOLDER = "web_vectorstore"

# === TEXT CLEANER ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

# === STREAMLIT UI ===
st.title("🌐 Web RAG: URL-based Document QA")
st.write("Enter URLs to scrape, process, and query.")

# === STEP 1: INPUT URLs ===
urls = st.text_area("Enter URLs (comma separated):", placeholder="https://example.com, https://wikipedia.org/...")
if st.button("Fetch & Process Data"):
    if urls:
        url_list = [u.strip() for u in urls.split(",")]
        st.write("🔍 Loading data from URLs...")
        
        # Load documents
        documents = []
        for url in url_list:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
            documents.extend(docs)
        
        st.success(f"✅ Loaded {len(documents)} documents.")
        
        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        st.write(f"📊 Created {len(chunks)} chunks.")
        
        # Embedding + FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(VECTOR_DB_FOLDER)
        st.success("✅ Data processed and stored in vector DB!")

# === STEP 2: QUESTION ANSWERING ===
if Path(f"{VECTOR_DB_FOLDER}/index.faiss").exists():
    st.subheader("💬 Ask Questions from URLs")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_DB_FOLDER, embeddings, allow_dangerous_deserialization=True)
    
    # Load lightweight model from HF Hub (FREE)
    with st.spinner("🚀 Loading language model..."):
        model_name = "tiiuae/falcon-7b-instruct"  # Free model, or use "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        local_llm = HuggingFacePipeline(pipeline=pipe)
    
    # Create RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm, retriever=db.as_retriever())
    
    user_query = st.text_input("Ask a question:")
    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_query)
        st.write("### ✅ Answer:")
        st.write(response)
else:
    st.info("⚠️ Please enter URLs and click 'Fetch & Process Data' first.")
