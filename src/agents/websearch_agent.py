import os
import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub  # <-- use free Hub API

# ---------- Setup ----------
load_dotenv()

VECTOR_DB_FOLDER = "web_vectorstore"  # local folder inside your project

def clean_text(text: str) -> str:
    """Basic cleaning: remove links, non-ascii, extra spaces"""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()

# ---------- UI ----------
st.title("🌐 Web RAG: URL-based Document QA")
st.write("Enter URLs to scrape, process, and query.")

urls = st.text_area(
    "Enter URLs (comma separated):",
    placeholder="https://example.com, https://wikipedia.org/..."
)

# ---------- Build Vector DB ----------
if st.button("Fetch & Process Data"):
    if not urls.strip():
        st.warning("Please paste at least one URL.")
        st.stop()

    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    st.write("🔍 Loading data from URLs...")

    documents = []
    for url in url_list:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for d in docs:
                d.page_content = clean_text(d.page_content)
            documents.extend(docs)
            st.success(f"Loaded: {url}")
        except Exception as e:
            st.error(f"Failed to load {url}: {e}")

    if not documents:
        st.error("No documents loaded. Check your URLs.")
        st.stop()

    st.success(f"✅ Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    st.write(f"📊 Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_FOLDER)
    st.success("✅ Data processed and stored in vector DB!")

# ---------- QA ----------
if Path(f"{VECTOR_DB_FOLDER}/index.faiss").exists():
    st.subheader("💬 Ask Questions from URLs")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(
        VECTOR_DB_FOLDER, embeddings, allow_dangerous_deserialization=True
    )

    with st.spinner("🚀 Loading language model..."):
        # ✅ Free Hugging Face Hub model (doesn't need token for public models)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",  # free text2text model
            model_kwargs={"temperature": 0.3, "max_new_tokens": 256}
        )

    retriever = db.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    user_query = st.text_input("Ask a question:")
    if user_query:
        with st.spinner("Thinking..."):
            try:
                answer = qa_chain.run(user_query)
                st.write("### ✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Query failed: {e}")
else:
    st.info("⚠️ Please enter URLs and click 'Fetch & Process Data' first.")
