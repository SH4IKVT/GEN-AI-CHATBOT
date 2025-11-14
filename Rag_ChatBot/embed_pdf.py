# embed_pdf.py (Modified to use .env)
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

load_dotenv()

# Config
# --- MODIFIED PART: Read PDF_PATH from .env ---
PDF_PATH = ("./Dsa.pdf") 
# --- END MODIFIED PART ---

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2:1.5b")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def generate_system_prompt(docs):
    """
    Generates a system prompt based on the initial content of the document.
    """
    # print(" Generating a system prompt for the document...")
    llm = OllamaLLM(model=LLM_MODEL)
    
    context_for_prompt_generation = "\n\n---\n\n".join([doc.page_content for doc in docs[:3]])
    
    prompt_template = f"""
Based on the following content from a PDF document, create a concise system prompt for a chatbot.
The prompt should instruct the chatbot to act as an expert on the document's main topics.
It should start with "You are an expert assistant..." and be no more than two sentences.

Document Content:
---
{context_for_prompt_generation}
---

Generated System Prompt:
"""
    
    system_prompt = llm.invoke(prompt_template)
    # print(f" Generated Prompt: '{system_prompt.strip()}'")
    return system_prompt.strip()

def main():
    # --- MODIFIED PART: Check if PDF_PATH is set and valid ---
    if not PDF_PATH:
        print("Error: PDF_PATH is not set in your .env file.")
        return

    if not os.path.exists(PDF_PATH):
        print(f"Error: File '{PDF_PATH}' not found. Check the path in your .env file.")
        return
    # --- END MODIFIED PART ---

    collection_name = os.path.splitext(os.path.basename(PDF_PATH))[0].lower().replace(" ", "_")
    print(f"Processing '{PDF_PATH}' into collection '{collection_name}'...")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)

    system_prompt = generate_system_prompt(chunks)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=DB_PATH,
        collection_metadata={"system_prompt": system_prompt}
    )

    print(f"\n PDF '{PDF_PATH}' embedded into ChromaDB collection '{collection_name}' at '{DB_PATH}'")

if __name__ == "__main__":
    main()
