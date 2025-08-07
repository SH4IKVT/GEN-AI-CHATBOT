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
from langchain import HuggingFaceHub

# --- CONFIG ---
VECTOR_DB_FOLDER = "web_vector_db"

# --- UI HEADER ---
st.title("🌐 Web-based RAG Chatbot (No Download, Online Model)")

# --- Get Hugging Face Token ---
hf_token = st.text_input("🔐 Enter your HuggingFace API Token:", type="password")
if not hf_token:
    st.warning("Please enter your HuggingFace token to continue.")
    st.stop()

# --- Enter URLs to Scrape ---
urls_input = st.text_area(
    "🌍 Enter URLs (comma separated):",
    placeholder="https://example.com, https://wikipedia.org/..."
)

# --- Web Scraping Logic ---
def scrape_urls(urls):
    all_text = ""
    for url in urls:
        try:
            res = requests.get(url.strip(), timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "header", "footer", "nav", "form", "aside"]):
                tag.decompose()

            # Extract and clean text
            text = ' '.join(soup.stripped_strings)
            text = re.sub(r'\s+', ' ', text)
            all_text += text + "\n"

        except Exception as e:
            st.warning(f"❌ Failed to fetch {url.strip()}: {e}")
    return all_text

# --- Button: Scrape and Embed ---
if st.button("🔍 Scrape & Embed Data"):
    if not urls_input.strip():
        st.error("⚠️ Please enter at least one URL.")
    else:
        urls = urls_input.split(",")
        scraped_text = scrape_urls(urls)

        # Split and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([scraped_text])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(VECTOR_DB_FOLDER)
        st.success("✅ Web data scraped and embedded into vector DB!")

# --- Retrieval + QA ---
if Path(f"{VECTOR_DB_FOLDER}/index.faiss").exists():
    st.subheader("💬 Ask a question from the scraped websites")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_DB_FOLDER, embeddings, allow_dangerous_deserialization=True)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Online model
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=hf_token  # ✅ Token passed here directly
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    user_query = st.text_input("📝 Your Question:")
    if user_query:
        with st.spinner("Thinking... 🤔"):
            answer = qa_chain.run(user_query)
        st.success("✅ Answer:")
        st.write(answer)
else:
    st.info("ℹ️ Please scrape and embed websites first to ask questions.")
