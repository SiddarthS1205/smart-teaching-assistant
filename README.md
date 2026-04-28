# 🎓 Smart Teaching Assistant

> A production-ready, full-stack academic AI assistant combining **RAG**, **MCP**, **Multi-Agent Systems**, **Guardrails**, **Observability**, and **Testing**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [MCP Tools](#mcp-tools)
- [Agent System](#agent-system)
- [Guardrails](#guardrails)
- [Observability](#observability)
- [API Reference](#api-reference)
- [Setup Instructions](#setup-instructions)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)

--- 

## Overview

The Smart Teaching Assistant lets students upload academic PDF documents and ask questions about them. The system uses **Retrieval-Augmented Generation (RAG)** to ground answers in the document content, a **Multi-Component Pipeline (MCP)** of modular tools, and a **multi-agent architecture** that routes queries to the most appropriate specialist agent.

**Key capabilities:**
- Upload PDF documents and process them through the full RAG pipeline
- Ask natural-language questions answered from document content
- Generate structured academic summaries
- Automatic routing between QA and Summarizer agents
- Safety guardrails that block harmful or irrelevant queries
- Full observability with structured logging and query tracking
- Comprehensive test suite (unit + integration)
- Modern chat-style UI with login page

---

## Architecture

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/3a7a497a-236b-4812-9791-03fb6763f306" />


---

## MCP Tools

Each tool is a self-contained module in `backend/tools/`:

| Tool | File | Responsibility |
|------|------|----------------|
| **DocumentLoader** | `document_loader.py` | Reads PDF files, extracts text using pypdf |
| **TextChunker** | `text_chunker.py` | Splits text into overlapping word-boundary chunks |
| **EmbeddingGenerator** | `embedding_generator.py` | Generates dense vectors via OpenAI `text-embedding-3-small` |
| **VectorDatabase** | `vector_database.py` | FAISS index for fast similarity search with persistence |
| **Retriever** | `retriever.py` | Embeds query and searches the vector store |
| **LLMGenerator** | `llm_generator.py` | Generates answers using `gpt-4o-mini` with RAG context |
| **Summarizer** | `summarizer.py` | Produces structured academic summaries |

### Pipeline Flow (Upload)

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/b3c8dcdb-4d08-4330-9577-2484cc7c6629" />


### Pipeline Flow (Query)

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/7632f3a0-d9df-409b-ac47-bee5840c3bde" />

---

## Agent System

Three agents in `backend/agents/`:

### 🔀 Router Agent
Decides which specialist agent handles the query using keyword matching:
- Keywords: `summary`, `summarize`, `overview`, `outline`, `brief`, `abstract`, `synopsis`, `recap`, `tldr` → **SummarizerAgent**
- Everything else → **QAAgent**

### 🤖 QA Agent
Answers questions using the full RAG pipeline:
1. Retrieves top-K relevant chunks from FAISS
2. Passes chunks + query to the LLM Generator
3. Returns a grounded answer

### 📝 Summarizer Agent
Generates a structured academic summary:
1. Samples chunks spread across the document
2. Calls the Summarizer Tool with the sampled content
3. Returns a structured summary with main topic, key concepts, findings, and conclusions

---

## Guardrails

Safety layer in `backend/guardrails/validator.py`:

### Query Validation
| Check | Rule |
|-------|------|
| Empty query | Rejected with `EMPTY_QUERY` |
| Too short | < 3 characters → `QUERY_TOO_SHORT` |
| Too long | > 1000 characters → `QUERY_TOO_LONG` |
| Harmful content | Regex patterns for hacking, weapons, drugs, etc. → `HARMFUL_CONTENT` |
| Irrelevant content | Gambling, MLM, etc. → `IRRELEVANT_CONTENT` |

### Upload Validation
| Check | Rule |
|-------|------|
| File type | Only `.pdf` accepted → `INVALID_FILE_TYPE` |
| Empty file | 0 bytes → `EMPTY_FILE` |
| File size | > 10 MB → `FILE_TOO_LARGE` |

---

## Observability

Structured logging in `backend/observability/logger.py`:

- **Structured logs** — timestamped, leveled, written to `logs/app.log`
- **Query tracking** — every query/response pair logged to `logs/queries.jsonl`
- **Request timing** — `@timed` decorator measures execution time of every tool/agent
- **In-memory log** — accessible via `GET /logs` endpoint
- **Error logging** — full stack traces on failures

---

## API Reference

### `POST /upload`
Upload and process a PDF document.

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "status": "success",
  "message": "Document 'paper.pdf' processed successfully.",
  "stats": {
    "filename": "paper.pdf",
    "pages": 12,
    "chunks": 48,
    "processing_time_ms": 3241.5
  }
}
```

---

### `GET /ask?q={query}`
Ask a question about the uploaded document.

**Response:**
```json
{
  "status": "success",
  "query": "What is the main argument?",
  "answer": "The main argument of the paper is...",
  "agent_used": "qa",
  "document": "paper.pdf",
  "response_time_ms": 1823.4
}
```

---

### `GET /summary`
Generate a structured summary of the uploaded document.

**Response:**
```json
{
  "status": "success",
  "summary": "## Main Topic\n...\n## Key Concepts\n...",
  "document": "paper.pdf",
  "response_time_ms": 4102.1
}
```

---

### `GET /status`
Get system status and document info.

### `GET /logs`
Get recent query logs (last 50 entries).

### `POST /login`
Authenticate with `{"username": "...", "password": "..."}`.

### `DELETE /document`
Clear the current document and reset the vector store.

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- OpenAI API key

### 1. Clone and navigate
```bash
git clone <repo-url>
cd smart-teaching-assistant
```

### 2. Create virtual environment
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 5. Start the backend

**PowerShell (Windows):**
```powershell
python main.py
```

**Or with uvicorn directly:**
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the frontend
Open `frontend/index.html` in your browser, or serve it:

**PowerShell (Windows):**
```powershell
# From project root
python -m http.server 3000 --directory frontend
# Then visit http://localhost:3000
```

**Demo credentials:** `admin` / `password123`

### 7. API Documentation
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Running Tests

```bash
cd backend

# Run all tests
pytest

# Run specific test files
pytest tests/test_tools.py -v
pytest tests/test_guardrails.py -v
pytest tests/test_agents.py -v
pytest tests/test_api.py -v

# Run with coverage
pip install pytest-cov
pytest --cov=. --cov-report=html
```

### Test Coverage

| Test File | What it tests |
|-----------|---------------|
| `test_tools.py` | Unit tests for all 7 MCP tools |
| `test_guardrails.py` | Unit tests for query and upload validators |
| `test_agents.py` | Unit tests for QA, Summarizer, and Router agents |
| `test_api.py` | Integration tests for all API endpoints |

---

## Project Structure

```
smart-teaching-assistant/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Settings from environment
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── .env.example
│   │
│   ├── tools/                     # MCP Tool modules
│   │   ├── __init__.py
│   │   ├── document_loader.py     # Tool 1: PDF reader
│   │   ├── text_chunker.py        # Tool 2: Text splitter
│   │   ├── embedding_generator.py # Tool 3: OpenAI embeddings
│   │   ├── vector_database.py     # Tool 4: FAISS store
│   │   ├── retriever.py           # Tool 5: Semantic search
│   │   ├── llm_generator.py       # Tool 6: Answer generation
│   │   └── summarizer.py          # Tool 7: Summarization
│   │
│   ├── agents/                    # Agentic system
│   │   ├── __init__.py
│   │   ├── base_agent.py          # Abstract base class
│   │   ├── qa_agent.py            # QA Agent (RAG)
│   │   ├── summarizer_agent.py    # Summarizer Agent
│   │   └── router_agent.py        # Router Agent
│   │
│   ├── guardrails/                # Safety layer
│   │   ├── __init__.py
│   │   └── validator.py           # Query + upload validators
│   │
│   ├── observability/             # Logging & monitoring
│   │   ├── __init__.py
│   │   └── logger.py              # Structured logger + query tracker
│   │
│   └── tests/                     # Test suite
│       ├── conftest.py
│       ├── test_tools.py          # Unit tests: MCP tools
│       ├── test_guardrails.py     # Unit tests: validators
│       ├── test_agents.py         # Unit tests: agents
│       └── test_api.py            # Integration tests: API
│
├── frontend/
│   ├── index.html                 # Login page
│   ├── app.html                   # Main chat application
│   ├── style.css                  # Complete stylesheet
│   └── app.js                     # Application logic
│
└── README.md
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | FAISS (CPU) |
| PDF parsing | PyPDF |
| Testing | pytest + httpx |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Logging | Python logging + JSONL |

---

## Academic Evaluation Checklist

- ✅ **RAG** — Document upload → chunk → embed → FAISS → retrieve → generate
- ✅ **MCP** — 7 modular, independently testable tools
- ✅ **Agentic System** — QA Agent, Summarizer Agent, Router Agent
- ✅ **Guardrails** — Query validation, upload validation, harmful content blocking
- ✅ **Observability** — Structured logging, query tracking, timing, `/logs` endpoint
- ✅ **Testing** — 215+ unit and integration tests ensuring system reliability
- ✅ **Login Page** — Session-based auth with demo credentials
- ✅ **Modern UI** — Chat-style interface with file upload, history, summary button
- ✅ **Full README** — Architecture, setup, API docs, tool/agent explanations


## 🎯 Motivation

This project demonstrates how modern AI systems can combine RAG, MCP, and multi-agent 
architectures to build scalable and reliable academic assistants.
