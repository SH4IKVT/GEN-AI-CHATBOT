# chat_app.py
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import chromadb

# ---------------------- CONFIG ----------------------
load_dotenv()
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2:1.5b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# ---------------------- FUNCTIONS ----------------------
def transform_query(llm, question, history):
    """Rewrite follow-up questions into standalone ones."""
    if not history:
        return question

    last_answer = history[-1]["answer"]
    prompt = f"""
Rewrite the following follow-up question to be a standalone question.
Use the previous answer to understand the user's intent.

Previous answer: {last_answer}
Follow-up question: {question}

Standalone question:
"""
    rewritten = llm.invoke(prompt)
    return rewritten.strip()


def chatting(llm, vectorstore, system_prompt, question, history):
    rewritten_question = transform_query(llm, question, history)
    results = vectorstore.similarity_search(rewritten_question, k=8)

    if not results:
        return "❌ No relevant context found in the PDF."

    context = "\n\n---\n\n".join([r.page_content for r in results])

    prompt = f"""
{system_prompt}
You must answer questions **only using the content of the PDF**.
Do not make up answers. If not found, say:
"I could not find the answer in the provided document."

Context from PDF:
{context}

Question:
{rewritten_question}
"""

    response = llm.invoke(prompt)
    history.append({"question": rewritten_question, "answer": response.strip()})
    return response.strip()

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="NBT Chatbot", layout="centered")
st.title("")

# Initialize models
with st.spinner("NBT chatbot is loading ..."):
    try:
        embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
        llm = OllamaLLM(model=LLM_MODEL)

        client = chromadb.PersistentClient(path=DB_PATH)
        collections = client.list_collections()

        if not collections:
            st.error(" No documents found in the database. Please run `embed_pdf.py` first.")
            st.stop()

        selected_collection_obj = collections[0]
        collection_name = selected_collection_obj.name
        metadata = selected_collection_obj.metadata
        system_prompt = metadata.get("system_prompt", "You are a helpful assistant.")

        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model,
            collection_name=collection_name
        )

        st.success(f"Connected to `{collection_name}` collection.")
    except Exception as e:
        st.error(f" Error connecting to database: {e}")
        st.stop()

# Maintain chat history in Streamlit
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat messages
for msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(msg["question"])
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])

# Chat input
user_query = st.chat_input("Ask a question from your document...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("💭 Thinking..."):
        response = chatting(llm, vectorstore, system_prompt, user_query, st.session_state.history)

    with st.chat_message("assistant"):
        st.markdown(response)
