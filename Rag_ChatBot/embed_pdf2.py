# embed_pdf.py
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# ---------- Load Environment Variables ----------
load_dotenv()

# ---------- Configuration ----------
PDF_PATH = "./Dsa.pdf"  # You can update this path if needed
DB_PATH = os.getenv("CHROMA_DB_PATH", "./vector_db/chroma_db")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2:1.5b")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------- Helper: System Prompt Generator ----------
def generate_system_prompt(docs):
    """
    Generates a concise system prompt that guides the chatbot
    based on the main topics in the PDF content.
    """
    llm = OllamaLLM(model=LLM_MODEL)

    context_for_prompt = "\n\n---\n\n".join([doc.page_content for doc in docs[:3]])

    prompt_template = f"""
Based on the following content from a PDF document, create a concise system prompt for a chatbot.
The prompt should instruct the chatbot to act as an expert on the document's main topics.
It should start with "You are an expert assistant..." and be no more than two sentences.

Document Content:
---
{context_for_prompt}
---

Generated System Prompt:
"""

    system_prompt = llm.invoke(prompt_template)
    return system_prompt.strip()


# ---------- Main Embedding Process ----------
def main():
    # Check if PDF file exists
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        print(f"❌ Error: File '{PDF_PATH}' not found. Please check the path.")
        return

    collection_name = os.path.splitext(os.path.basename(PDF_PATH))[0].lower().replace(" ", "_")
    print(f"📄 Processing '{PDF_PATH}' into collection '{collection_name}'...")

    # Load and split PDF
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    # Create embedding model
    embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)

    # Generate a system prompt dynamically from PDF content
    system_prompt = generate_system_prompt(chunks)

    # Store document embeddings in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=DB_PATH,
        collection_metadata={"system_prompt": system_prompt}
    )

    print(f"\n✅ PDF '{PDF_PATH}' embedded successfully into collection '{collection_name}'")
    print(f"📂 Stored at: {DB_PATH}")


if __name__ == "__main__":
    main()
