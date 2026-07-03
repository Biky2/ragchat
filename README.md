# RAGChat — Enterprise Intelligent RAG-Based Chatbot

Production-ready RAG chatbot with hybrid retrieval (dense vector + BM25 sparse), grounded answers with citations, document ingestion, and a full SaaS-style UI.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Browser (index.html)                          │
│         Upload docs · Stream chat · Citations · Session analytics       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP / SSE
┌─────────────────────────────────▼───────────────────────────────────────┐
│                         FastAPI (api/main.py)                           │
│  /ingest · /chat · /chat/stream · /documents · /health · /feedback    │
└──────┬──────────────┬──────────────┬──────────────┬─────────────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌────────────┐ ┌─────────────┐ ┌──────────┐ ┌─────────────────────────┐
│ Ingestion  │ │  Retrieval  │ │Generation│ │ Memory                  │
│ load/chunk │ │ dense+BM25  │ │ RAG chain│ │ PostgreSQL + sessions   │
│ embed      │ │ RRF hybrid  │ │ citations│ │ chat_history, docs      │
└─────┬──────┘ └──────┬──────┘ └────┬─────┘ └─────────────────────────┘
      │               │             │
      ▼               ▼             ▼
┌──────────────┐ ┌──────────┐ ┌─────────────┐
│ Vector Store │ │ BM25 idx │ │ LLM Layer   │
│ Pinecone /   │ │ in-memory│ │ Ollama / HF │
│ FAISS fallback│ │         │ │ fallback    │
└──────────────┘ └──────────┘ └─────────────┘
```

## Prerequisites

- Python 3.11+
- PostgreSQL 16 (via Docker or local install)
- [Ollama](https://ollama.ai) with `llama3.2` model (optional — HF fallback available)
- [Pinecone](https://www.pinecone.io) free tier API key (optional — FAISS fallback built-in)

## Quick Start

### 1. Start PostgreSQL (and Redis)

```bash
cd ragchat
docker-compose up -d
```

### 2. Configure environment

```bash
cp .env .env.local   # edit as needed
```

Minimum required for local dev (FAISS + Ollama, no Pinecone):

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ragchat
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
```

### 3. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 4. Pull Ollama model

```bash
ollama pull llama3.2
```

### 5. Run the server

```bash
cd ragchat
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000

## Pinecone Free Tier Setup

1. Sign up at https://app.pinecone.io
2. Create a project and copy your API key
3. Set in `.env`:
   ```env
   PINECONE_API_KEY=your_key_here
   PINECONE_INDEX_NAME=ragchat
   PINECONE_ENVIRONMENT=us-east-1-aws
   ```
4. The index is auto-created on first startup (384 dimensions, cosine metric)

If Pinecone is unavailable, the system silently falls back to a local FAISS index at `./data/faiss_index`.

## Running Tests

```bash
cd ragchat
pytest tests/ -v
```

## API Examples (curl)

### Health check

```bash
curl http://localhost:8000/health
```

### Ingest a file

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf" \
  -F "strategy=fixed"
```

### Ingest raw text

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here.", "source_name": "notes", "strategy": "sentence"}'
```

### Chat (non-streaming)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "session_id": "my-session-1", "top_k": 5}'
```

### Chat (SSE streaming)

```bash
curl -N "http://localhost:8000/chat/stream/my-session-1?query=Summarize%20the%20document&top_k=5"
```

### List documents

```bash
curl http://localhost:8000/documents
```

### Delete document

```bash
curl -X DELETE http://localhost:8000/documents/{document_id}
```

### Session history

```bash
curl http://localhost:8000/history/my-session-1
```

### Submit feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"chat_history_id": "uuid-here", "rating": 1, "comment": "Helpful answer"}'
```

## Project Structure

```
ragchat/
├── api/           FastAPI app, routes, schemas
├── config/        pydantic-settings
├── core/          LLM + vector store abstractions
├── ingestion/     load, chunk, embed, ingest pipeline
├── retrieval/     dense, sparse, hybrid RRF
├── generation/    prompt builder, chain, citations
├── memory/        PostgreSQL + session store
├── static/        index.html frontend
├── tests/         pytest suite
└── data/          FAISS index (gitignored)
```

## Features

- **Hybrid retrieval**: Pinecone/FAISS dense search + BM25 sparse, fused via RRF (k=60)
- **Chunking strategies**: fixed, sentence, semantic
- **Local embeddings**: sentence-transformers `all-MiniLM-L6-v2`
- **LLM fallback**: Ollama primary → HuggingFace Inference API
- **Conversation memory**: last 3 exchanges prepended to queries
- **Streaming**: SSE token-by-token responses
- **Citations**: `[Source N]` parsing mapped to chunk metadata
- **Admin**: document list, delete, ingest status tracking

## License

MIT
