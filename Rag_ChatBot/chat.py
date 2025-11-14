# chat.py
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import chromadb

load_dotenv()

# Config
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2:1.5b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

history = []

def transform_query(llm, question):
    """Rewrite follow-up questions into standalone questions using last history."""
    if not history:
        return question
        
    last_answer = history[-1]["answer"]
    prompt = f"""
Rewrite the following follow-up question to be a standalone question.
You must use the context from the previous answer to fully understand the user's intent.

Previous answer: {last_answer}
Follow-up question: {question}

Standalone question:
"""
    rewritten = llm.invoke(prompt)
    return rewritten.strip()

def chatting(llm, vectorstore, system_prompt, question):
    global history
    rewritten_question = transform_query(llm, question)

    # Search top 5 relevant chunks from PDF
    results = vectorstore.similarity_search(rewritten_question, k=8)

    if not results:
        print("No relevant context found in the PDF.")
        return

    # Combine retrieved context
    context = "\n\n---\n\n".join([r.page_content for r in results])

    # print(f"--- RETRIEVED CONTEXT ---\n{context}\n--------------------------")

    # Use the DYNAMIC system prompt fetched from metadata
    prompt = f"""
{system_prompt}
You must answer questions **only using the content of the PDF documents provided**.
Do not make up answers. If the answer is not in the context, reply:
"I could not find the answer in the provided document."

Context from PDF:
{context}

Question:
{rewritten_question}
"""

    response = llm.invoke(prompt)
    print("\nAssistant:", response.strip(), "\n")

    # Save history for multi-turn conversations
    history.append({"question": rewritten_question, "answer": response.strip()})

def main():
    # Initialize models
    embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
    llm = OllamaLLM(model=LLM_MODEL)

    # Connect to ChromaDB and list available collections
    client = chromadb.PersistentClient(path=DB_PATH)
    collections = client.list_collections()

    # --- MODIFIED SECTION ---
    # Automatically select the first document without asking
    if not collections:
        print("Error: No documents found in the database. Please run embed_pdf.py first.")
        return

    # Automatically select the first available document collection
    selected_collection_obj = collections[0]
    collection_name = selected_collection_obj.name
    # --- END MODIFIED SECTION ---

    # Load the selected vectorstore
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    
    # Retrieve the dynamic system prompt from the collection's metadata
    metadata = selected_collection_obj.metadata
    system_prompt = metadata.get("system_prompt", "You are a helpful assistant.") # Fallback prompt

    print(f"\n Automatically connected to '{collection_name}'.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    # Start the chat loop
    while True:
        question = input(f"Ask from pdf --> ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        chatting(llm, vectorstore, system_prompt, question)

if __name__ == "__main__":
    main()