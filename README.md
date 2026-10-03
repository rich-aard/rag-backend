# RAG Backend

A FastAPI backend with two REST APIs:

1. **Document Ingestion API** – upload a `.pdf` or `.txt` file, split it with one of two chunking strategies, embed the chunks, store them in Qdrant, and save the document metadata in SQLite.
2. **Conversational RAG API** – ask questions about the uploaded documents across multiple turns, and book interviews through the same chat. Chat history lives in Redis.

The RAG pipeline is written by hand in `src/services/rag.py`. It does **not** use `RetrievalQAChain` or any other prebuilt retrieval chain.

## Tech stack

| Part                  | Choice                                                    | Why                                                         |
| --------------------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| API                   | FastAPI                                                   | Typed request/response models and automatic docs at `/docs` |
| Vector store          | Qdrant                                                    | Runs locally with one Docker container, no account needed   |
| Chat memory           | Redis                                                     | Fast per-session history with an automatic expiry (TTL)     |
| Metadata and bookings | SQLite (SQLAlchemy)                                       | No extra service to run; fits small tabular data            |
| LLM                   | Groq (`openai/gpt-oss-120b`) via `langchain-groq`         | Hosted model, only an API key is needed                     |
| Embeddings            | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) | Small, runs on CPU                                          |
| Chunking              | LangChain text splitters                                  | Two ready-made strategies                                   |

## How it works

**Ingestion** (`POST /api/v1/documents`)

```
upload -> extract text -> split into chunks -> embed -> store vectors in Qdrant
                                                     -> save metadata row in SQLite
```

**Chat** (`POST /api/v1/chat`)

```
load history from Redis
  -> is this a booking request? (LLM extracts name/email/date/time)
       yes -> validate the details, ask for missing ones, or save the booking
       no  -> rewrite the question to stand alone (uses recent history)
              -> search Qdrant (top 5 chunks)
              -> build the prompt (context + history + question)
              -> LLM answers
save the turn to Redis
```

### Chunking strategies

Chosen with the `chunking_strategy` form field:

| Value       | Splitter                         | Size / overlap        |
| ----------- | -------------------------------- | --------------------- |
| `recursive` | `RecursiveCharacterTextSplitter` | 1000 / 200 characters |
| `token`     | `TokenTextSplitter`              | 256 / 50 tokens       |

### Interview booking

The LLM reads the current message and the recent conversation and extracts name, email, interview_date and interview_time into a structured object. The code then validates them: email format, date and time format, and that the date is not in the past. If something is missing, the assistant asks for it, and details can arrive over several messages. When everything is present, the booking is saved to SQLite.

## Project structure

```
src/
  main.py              # app setup, startup (loads models, connects services)
  api/                 # routes: ingestion_api, chat_api, health, dependencies
  services/            # extraction, chunking, embeddings, vector_store,
                       # ingestion, rag, booking, memory, llm, prompts
  db/                  # SQLAlchemy models, session, repositories
  schemas/             # Pydantic request/response models
  core/                # settings and custom exceptions
docker-compose.yml     # Qdrant and Redis
```

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, and a free [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/rich-aard/rag-backend
cd rag-backend

# 1. Start Qdrant and Redis
docker compose up -d

# 2. Install dependencies
uv sync

# 3. Configure
cp .env.example .env        # on Windows PowerShell: copy .env.example .env
# open .env and set GROQ_API_KEY

# 4. Run
uv run uvicorn src.main:app --reload
```

The first install is large (PyTorch comes with `sentence-transformers`), and the first start downloads the embedding model (about 90 MB), so expect a few minutes. The first use of the `token` strategy also downloads a small tokenizer file, so it needs internet access.

When the server is up:

- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Qdrant dashboard: http://localhost:6333/dashboard

## Usage

The easiest way to try everything is the Swagger UI at `/docs`. The same calls with `curl` are below. On Windows PowerShell, use `curl.exe` instead of `curl`.

### Upload a document

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -F "file=@sample.pdf" \
  -F "chunking_strategy=recursive"
```

Response (`201`):

```json
{
  "doc_id": "3f2b8c1e-...",
  "filename": "sample.pdf",
  "file_type": "pdf",
  "chunking_strategy": "recursive",
  "chunk_count": 42,
  "created_at": "2026-10-03T06:58:47.145709"
}
```

Errors: `415` unsupported file type, `413` file too large, `422` no extractable text or a corrupted file, `503` vector store unavailable.

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo1", "question": "What is QLoRA?"}'
```

Reuse the same `session_id` for follow-ups such as `"How do we do it?"`. The previous turns are read from Redis and used to resolve what "it" means.

Response:

```json
{
  "kind": "rag",
  "answer": "...",
  "sources": [
    {
      "doc_id": "...",
      "filename": "sample.pdf",
      "chunk_index": 12,
      "text": "...",
      "score": 0.64
    }
  ],
  "booking": null
}
```

### Book an interview

Use the same endpoint and send the details in one message or across several:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo2", "question": "I would like to book an interview"}'

curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo2", "question": "Jane Doe, jane@example.com, tomorrow at 3pm"}'
```

When all four details are present, the response has `"kind": "booking"` and a `booking` object with `"status": "confirmed"`. The booking is stored in the `bookings` table in `data/app.db`.

## Configuration

Set in `.env`. Only `GROQ_API_KEY` is required.

| Variable       | Default                    | Meaning                                                                                           |
| -------------- | -------------------------- | ------------------------------------------------------------------------------------------------- |
| `GROQ_API_KEY` | –                          | Groq API key (required)                                                                           |
| `QDRANT_URL`   | `http://localhost:6333`    | Qdrant address                                                                                    |
| `REDIS_URL`    | `redis://localhost:6379/0` | Redis address                                                                                     |
| `TIMEZONE`     | `UTC`                      | IANA timezone (for example `Asia/Kathmandu`) used to interpret "today" and "tomorrow" in bookings |

Other settings (model names, chunk sizes, history length, file size limit) have defaults in `src/core/configs.py` and can be overridden with environment variables of the same name in upper case. Defaults worth knowing: upload limit 50 MB, 5 chunks retrieved per question, the last 20 messages kept per session for one hour.

## Design notes

- **One FastAPI app, two routers.** Both APIs share the same configuration, embedding model and vector store.
- **Models load once at startup.** The embedding model, LLM client, vector store and Redis connection are created in the app lifespan and injected into routes. The embedding dimension is checked at startup, so a model/config mismatch fails immediately.
- **Layers.** Routes handle HTTP only, services hold the logic, repositories hold the database access, and Pydantic schemas define the API shapes.
- **Custom exceptions** (`ExtractionError`, `VectorStoreError`, `MemoryStoreError`, `LLMError`) are raised in services and mapped to HTTP status codes in the routes.
- **Question rewriting.** Follow-up questions are rewritten into standalone ones before searching, so retrieval has real keywords to match.

## Limitations

- Booking details are re-read from the chat history on each turn (the last 20 messages). There is no separate booking state.
- Existing bookings cannot be changed, cancelled or looked up. Those requests are declined.
- The date check compares dates only, so a time earlier today is not rejected.
- If the database write fails after the vectors are stored, the vectors stay in Qdrant without a metadata row.
- `created_at` timestamps are stored in UTC. `TIMEZONE` only affects how booking dates are interpreted.
- Tables are created on startup but not migrated. After changing a model, delete the affected table (or `data/app.db`) and restart.
- Scanned (image-only) PDFs are rejected because there is no OCR.
- Qdrant returns the closest chunks even when none are relevant, so the `sources` list can contain weak matches. The prompt tells the model to say it doesn't know in that case.

## Development checks

```bash
uv run ruff check .
uv run ruff format .
uv run mypy --strict src
```

To reset everything: `docker compose down -v` and delete `data/app.db`.
