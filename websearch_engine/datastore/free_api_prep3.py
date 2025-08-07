import os
import streamlit as st
from pathlib import Path
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub

VECTOR_DB_FOLDER = "web_vector_db"
st.title("🌐 Smart Web RAG Chatbot")

hf_token = st.text_input("🔑 Enter your HuggingFace API Token:", type="password")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

urls_input = st.text_area("🌍 Enter URLs (comma separated):", placeholder="https://example.com, https://wikipedia.org")

def scrape_urls(urls):
    all_text = ""
    for url in urls:
        try:
            article = Article(url.strip())
            article.download()
            article.parse()
            all_text += article.text + "\n"
        except Exception as e:
            st.warning(f"❌ Failed to fetch {url}: {e}")
    return all_text

if st.button("🔎 Scrape & Embed"):
    if not urls_input:
        st.error("Please enter at least one URL.")
    else:
        urls = urls_input.split(",")
        raw_text = scrape_urls(urls)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([raw_text])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(VECTOR_DB_FOLDER)
        st.success("✅ Web content embedded and saved successfully!")

if Path(f"{VECTOR_DB_FOLDER}/index.faiss").exists():
    st.subheader("💬 Ask a question based on scraped web data")
    user_query = st.text_input("Ask:")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_DB_FOLDER, embeddings, allow_dangerous_deserialization=True)

    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.3, "max_length": 512})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    if user_query:
        with st.spinner("🤖 Generating answer..."):
            response = qa_chain.run(user_query)
        st.success("✅ Answer:")
        st.write(response)
else:
    st.info("📌 Please scrape and embed some URLs first.")
# pip install lxml[html_clean]
# pip install newspaper3k