# PDF Chatbot with LangChain, Ollama, and ChromaDB

This project allows you to embed the content of a PDF into a vector database (ChromaDB) and chat with it using an AI assistant powered by Ollama. The chatbot answers **strictly based on the PDF content**.

---

## **Features**

- Convert any PDF into searchable chunks and store embeddings in ChromaDB.
- Automatically generate a system prompt based on the document's content.
- Multi-turn chat with follow-up question understanding.
- No external knowledge—answers come only from the PDF.

---

## **Project Structure**

.
├── .env # Environment variables
├── embed_pdf.py # Script to embed a PDF into ChromaDB
├── chat.py # Script to chat with the PDF
├── requirements.txt # Python dependencies
└── Dsa.pdf # Example PDF (replace with your own)


---

## **Setup Instructions**

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>


pip install -r requirements.txt


3. Configure environment variables

Create a file named .env in the root folder with the following content:

# Path to your PDF file
PDF_PATH=./Dsa.pdf

# Directory to store ChromaDB embeddings
CHROMA_DB_PATH=./chroma_db

# Ollama embedding model
OLLAMA_EMBED_MODEL=mxbai-embed-large

# Ollama language model for generating answers
OLLAMA_LLM_MODEL=qwen2:1.5b

You can replace Dsa.pdf with your own PDF and change the models if needed.

4. Embed PDF

Run the embedding script to process the PDF and store embeddings:

python embed_pdf.py


This will:

Split the PDF into chunks.

Generate embeddings using Ollama.

Create a system prompt from the PDF content.

Store everything in ChromaDB.



5. Chat with the PDF

Start the chat interface:

python chat.py


The first PDF in the database is automatically selected.

Ask questions based on the PDF content.

Type exit or quit to end the chat session.