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
from langchain_huggingface import HuggingFaceEndpoint

# ---------- Setup ----------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Add HUGGINGFACEHUB_API_TOKEN to your .env")
    st.stop()

VECTOR_DB_FOLDER = "web_vectorstore"

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()

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
        # Important bits:
        # 1) Use task='text2text-generation' for FLAN
        # 2) Use max_new_tokens instead of max_length
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            task="text2text-generation",
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.95,
            return_full_text=False,
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
