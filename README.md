# 📊 FinWiz - Corporate 10-K Q&A RAG System

A lightweight Retrieval-Augmented Generation (RAG) project that lets querying a company's 10-K PDF using Gemini (Google GenAI SDK).

## 🧠 Overview
This project demonstrates how to:
- Parse a 10-K PDF
- Chunk and embed text using Google GenAI embeddings
- Store embeddings in a FAISS index
- Use Gemini for context-aware Q&A

## ⚙️ Setup

### 1. Environment
```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your_api_key_here"
```

### 2. Run
```bash
python app.py
```

### 3. Ask questions interactively
```
📄 Reading PDF from: docs/nvidia_10K.pdf
🔹 Generating embeddings for 716 chunks...
✅ FAISS index and chunks saved in cache

❓ Ask a question (or type 'exit'): What is the latest annual revenue of Nvidia?

💡 Answer:
The latest annual revenue of Nvidia is $130,497 million for the year ended January 26, 2025.
```

## 🧩 Tech Stack
- **Google GenAI SDK** – for embeddings + Gemini Q&A
- **FAISS** – for efficient vector similarity search
- **pypdf** – for text extraction
- **Python 3.11+**

## 📁 Project Structure
```
finwiz/
├── app.py
├── rag_system.py
├── requirements.txt
├── dockerfile
├── docker-compose.yml
├── README.md
└── cache/
```

## 🐳 Docker
```bash
docker build -t finwiz .
docker run -e GOOGLE_API_KEY=<your_api_key> -it finwiz bash
python app.py
```

## 🧠 Notes
- The FAISS index is cached to `cache/` for faster reuse.
- Ensure you use the latest `google-genai` (>=1.46.0).
