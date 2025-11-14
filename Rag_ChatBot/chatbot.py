# chat_app.py
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import chromadb

# ========================= CONFIG =========================
load_dotenv()
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2:1.5b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# ========================= FUNCTIONS =========================
def transform_query(llm, question, history):
    """Rewrite follow-up question into standalone using context."""
    if not history:
        return question

    last_answer = history[-1]["answer"]
    prompt = f"""
Rewrite the following follow-up question to be standalone.
Use this context:

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
        return "⚠️ Sorry, I couldn’t find relevant context in the document."

    context = "\n\n---\n\n".join([r.page_content for r in results])

    prompt = f"""
{system_prompt}
You must answer strictly based on the PDF content.
If unsure, say: "I could not find the answer in the provided document."

Context:
{context}

Question:
{rewritten_question}
"""

    response = llm.invoke(prompt)
    history.append({"question": rewritten_question, "answer": response.strip()})
    return response.strip()

# ========================= UI SETUP =========================
st.set_page_config(
    page_title="The Next Big Tech Chatbot Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 0.3em;
        color: #00b4d8;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #adb5bd;
        margin-bottom: 2em;
    }
    .stChatMessage {
        background: #1a1d23 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    .stChatInput input {
        border-radius: 10px !important;
        background-color: #121417 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🤖 The Next Big Tech Chatbot Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your smart AI partner for document-based intelligence</div>", unsafe_allow_html=True)

# ========================= LOAD MODELS =========================
with st.spinner("🚀 Initializing AI systems... please wait..."):
    try:
        embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
        llm = OllamaLLM(model=LLM_MODEL)

        client = chromadb.PersistentClient(path=DB_PATH)
        collections = client.list_collections()

        if not collections:
            st.error("⚠️ No documents found in your database. Please run `embed_pdf.py` first.")
            st.stop()

        # Let user choose which collection to chat with
        collection_names = [c.name for c in collections]
        selected_collection_name = st.selectbox("📚 Select Document Collection", collection_names)
        selected_collection_obj = next(c for c in collections if c.name == selected_collection_name)

        metadata = selected_collection_obj.metadata
        system_prompt = metadata.get("system_prompt", "You are a helpful assistant.")

        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model,
            collection_name=selected_collection_name
        )

        st.success(f"✅ Connected to `{selected_collection_name}` collection successfully.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# ========================= CHAT LOGIC =========================
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous messages
for msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {msg['question']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Next Big Tech:** {msg['answer']}")

# Input box
user_query = st.chat_input("💬 Ask a question from your uploaded document...")

if user_query:
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_query}")

    with st.spinner("🤔 Thinking..."):
        response = chatting(llm, vectorstore, system_prompt, user_query, st.session_state.history)

    with st.chat_message("assistant"):
        st.markdown(f"**Next Big Tech:** {response}")
