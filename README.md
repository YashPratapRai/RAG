# 🔍 RAG System — Document Question Answering with Retrieval-Augmented Generation

> Ask questions from your documents and get accurate, context-grounded answers — no hallucinations.

---

## 🧠 What This Project Does

Large Language Models are powerful, but they hallucinate when they lack access to your specific data. This project solves that with **Retrieval-Augmented Generation (RAG)** — a technique that grounds LLM responses in real retrieved content before generating an answer.

**In plain terms:** Upload a document → ask a question → get an accurate, source-backed answer.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Embedding Generation         ← Query converted to vector
    │
    ▼
Vector Similarity Search     ← FAISS finds most relevant chunks
    │
    ▼
Document Chunk Retrieval     ← Top-k relevant passages extracted
    │
    ▼
LLM Response Generation      ← Answer generated using retrieved context
```

---

## 📂 Project Structure

```
RAG/
├── agentic_rag/             # Experimental: Agentic AI workflow with LangGraph
├── data/                    # Document ingestion & text chunking utilities
├── notebook/
│   └── document.ipynb       # Full RAG pipeline walkthrough (start here)
├── main.py                  # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline Components

### 1. Document Ingestion
Loads and preprocesses documents — handles text extraction and cleaning before any embeddings are generated.

### 2. Text Chunking
Splits large documents into smaller, overlapping chunks to:
- Improve retrieval precision
- Preserve local context across chunk boundaries
- Stay within LLM token limits

### 3. Embeddings
Each chunk is encoded into a dense vector using an embedding model, enabling **semantic search** rather than brittle keyword matching.

### 4. Vector Store (FAISS)
Embeddings are stored in a FAISS index for fast approximate nearest-neighbor search. At query time, the most relevant chunks are retrieved in milliseconds.

### 5. LLM Response Generation
Retrieved chunks are injected into the LLM prompt as context. The model generates an answer **grounded in your documents**, dramatically reducing hallucinations.

---

## 🤖 Bonus: Agentic AI Exploration (`agentic_rag/`)

A separate experimental module exploring **agent-based reasoning** using LangGraph. The agent can:
- Decide whether to retrieve or answer directly
- Chain retrieval → reasoning → response steps
- Reflect and retry if the first retrieval isn't sufficient

> This is an exploration, not production-ready code — but it demonstrates how RAG can evolve into a more autonomous system.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM | LangChain |
| Vector Database | FAISS |
| Agentic Workflow | LangGraph |
| Interface | Python / Jupyter Notebook |
| Language | Python 3.x |

---

## ▶️ Quickstart

**1. Clone the repo**
```bash
git clone https://github.com/YashPratapRai/RAG.git
cd RAG
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the pipeline**
```bash
python main.py
```

**4. Or explore interactively**
```bash
jupyter notebook notebook/document.ipynb
```

---


## 👨‍💻 Author
Yash Pratap Rai

**Yash Pratap Rai** — CS student focused on Generative AI, RAG systems, and Agentic AI.

[![GitHub](https://img.shields.io/badge/GitHub-YashPratapRai-181717?logo=github)](https://github.com/YashPratapRai/RAG)
