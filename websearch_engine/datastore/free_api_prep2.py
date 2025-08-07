import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- CONFIG ---
VECTOR_DB_FOLDER = "web_vector_db"

# --- STREAMLIT UI ---
st.title("🌐 Web-based RAG Chatbot (Free + Offline)")

urls_input = st.text_area(
    "Enter URLs (comma separated):",
    placeholder="https://example.com, https://en.wikipedia.org/wiki/Natural_language_processing"
)


# --- STEP 1: Scrape & clean ---
def scrape_urls(urls):
    all_text = ""
    for url in urls:
        try:
            res = requests.get(url.strip(), timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = ' '.join(soup.stripped_strings)
            text = re.sub(r'\s+', ' ', text)
            all_text += text + "\n"
        except Exception as e:
            st.warning(f"Failed to fetch {url}: {e}")
    return all_text


# --- STEP 2: Load lightweight open-source LLM ---
@st.cache_resource
def load_local_llm():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)


# --- Scrape & Embed ---
if st.button("Scrape & Embed"):
    if not urls_input:
        st.error("Please enter at least one URL.")
    else:
        urls = urls_input.split(",")
        text_data = scrape_urls(urls)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([text_data])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(VECTOR_DB_FOLDER)
        st.success("✅ Data embedded and saved!")


# --- QA Section ---
if Path(f"{VECTOR_DB_FOLDER}/index.faiss").exists():
    st.subheader("💬 Ask a question from the scraped data")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_DB_FOLDER, embeddings, allow_dangerous_deserialization=True)

    llm = load_local_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    user_query = st.text_input("Your Question:")
    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_query)
        st.success("✅ Answer:")
        st.write(response)
else:
    st.info("ℹ️ Please scrape and embed URLs first.")
