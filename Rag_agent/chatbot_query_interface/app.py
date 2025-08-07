import os
import re
from pathlib import Path
import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import fitz  # PyMuPDF

# --- CONFIG ---
VECTOR_DB_FOLDER = "my_vector_db"
FIGURE_OUTPUT = "extracted_figures"

# --- STEP 1: Auto-detect local file ---
def detect_file():
    for ext in ['.pdf', '.txt', '.docx']:
        path = Path(f"data{ext}")
        if path.exists():
            return str(path), ext[1:]
    return None, None

# --- STEP 2: Load file ---
def load_document(file_path, file_type):
    path = str(Path(file_path).resolve())
    if file_type == 'txt':
        loader = TextLoader(path)
    elif file_type == 'pdf':
        loader = PyPDFLoader(path)
    elif file_type == 'docx':
        loader = Docx2txtLoader(path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

# --- STEP 3: Clean text ---
def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

# --- STEP 4: Add metadata ---
def attach_metadata(documents, file_path):
    for doc in documents:
        doc.metadata = {"source": Path(file_path).name}
    return documents

# --- STEP 5: Chunking ---
def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# --- STEP 6: Extract images if PDF ---
def extract_images_from_pdf(pdf_path, output_folder):
    Path(output_folder).mkdir(exist_ok=True)
    doc = fitz.open(pdf_path)
    count = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            filename = f"{output_folder}/page{page_num+1}_img{img_index+1}.{image_ext}"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            count += 1
    return count

# --- STEP 7: Vector embedding ---
def embed_and_store(chunks, db_folder):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_folder)

# --- STREAMLIT UI ---
st.title("📚 Local File Processor with RAG")

file_path, file_type = detect_file()

if file_path:
    st.success(f"Found file: `{file_path}`")
    documents = load_document(file_path, file_type)
    
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    documents = attach_metadata(documents, file_path)
    chunks = chunk_documents(documents)

    st.write(f"✅ Loaded {len(documents)} document(s)")
    st.write(f"📊 Created {len(chunks)} chunks")

    if st.button("Embed & Save to FAISS"):
        embed_and_store(chunks, VECTOR_DB_FOLDER)
        st.success("✅ Embedding complete & saved!")

    if file_type == "pdf":
        if st.button("Extract Images from PDF"):
            count = extract_images_from_pdf(file_path, FIGURE_OUTPUT)
            st.success(f"🖼️ Extracted {count} image(s) to `{FIGURE_OUTPUT}` folder.")
else:
    st.error("❌ No `data.txt`, `data.pdf`, or `data.docx` file found in the current directory.")

from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Load vector DB if exists
if Path(f"{VECTOR_DB_FOLDER}/index.faiss").exists():
    st.subheader("💬 Ask Questions from Your Document")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_DB_FOLDER, embeddings, allow_dangerous_deserialization=True)

    # Load Mistral-7B (or any HF model)
    with st.spinner("🚀 Loading Mistral-7B model... (may take 30s+)"):
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        local_llm = HuggingFacePipeline(pipeline=pipe)

    # LangChain QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm, retriever=db.as_retriever())

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_query = st.chat_input("Ask a question...")

    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_query)
        st.session_state.messages.append(("user", user_query))
        st.session_state.messages.append(("bot", response))

    for role, msg in st.session_state.messages:
        st.chat_message(role).write(msg)

else:
    st.warning("⚠️ Please embed documents into vector DB first using the 'Embed & Save' button above.")

# mistral token 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