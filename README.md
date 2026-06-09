# Agentic AI Product Discovery

A conversational e-commerce search assistant combining hybrid retrieval, tool-using agents, and in-session checkout , built as the research prototype backing an M.Sc. thesis at Berliner Hochschule für Technik.

---

## What it does

Users describe what they want in plain language. The agent searches a product catalogue using hybrid BM25 + dense retrieval, presents ranked results, and can add items to a cart and complete checkout , all inside the same conversation turn.

---

## Architecture

```
User ──► FastAPI  ──► LangGraph Agent (Gemini)
                          │
              ┌───────────┼────────────────┐
              ▼           ▼                ▼
        search_products  ucp_add_to_cart  ucp_checkout
              │
              ▼
        OpenSearch (HNSW/FAISS)
        BM25 + dense vector (0.4 / 0.6)
        BAAI/bge-small-en-v1.5 embeddings
```

**Services**

| Path | Role |
|---|---|
| `services/api/` | FastAPI app , search, chat, cart, health endpoints |
| `services/ingestion/` | One-shot ETL: HuggingFace dataset → OpenSearch |
| `shared/config/` | Pydantic settings, shared across services |
| `demo.html` | Static frontend wired to the API |

---

## Quickstart

```bash
# 1. Copy and fill in your keys
cp .env.example .env

# 2. Start OpenSearch
docker compose up -d opensearch

# 3. Ingest the product catalogue
cd services && python -m ingestion

# 4. Start the API
uvicorn api.main:app --reload --port 8000

# 5. Open demo.html in your browser
```

**Required env vars:** `GOOGLE_API_KEY`, `OPENSEARCH_HOST`  
**Optional:** `AGENT_MODE=full_commerce` to enable cart and checkout endpoints (default: on)

---

## API surface

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search` | Hybrid keyword + vector search with filters |
| `POST` | `/nl-search` | LLM-parsed natural language search |
| `GET` | `/suggest` | Autocomplete prefix suggestions |
| `POST` | `/chat` | Multi-turn agentic conversation |
| `POST` | `/cart/session` | Create UCP checkout session |
| `POST` | `/cart/checkout` | Complete order (mock payment) |
| `GET` | `/health` | Liveness + OpenSearch status |

Full schema at `/docs` once the server is running.

---

## Research roadmap

This prototype is the starting point for a three-phase research programme:

1. **Multimodal queries** , extend retrieval to accept image + text inputs (CLIP-style joint embeddings fused with the existing BM25/dense pipeline)
2. **Implicit signal capture** , instrument the agent's tool calls as a structured event stream and use session-level behavioral signals to re-rank results within the same conversation
3. **Benchmark evaluation** , measure against Amazon ESCI (~130k queries, NDCG) and SQID (~9k multimodal queries) for retrieval quality, and the Amazon Reviews 2023 interaction graph for session re-ranking (Recall@K)

---

## Stack

Python 3.11 · FastAPI · LangGraph · LangChain · Gemini 2.0 Flash · OpenSearch 2.x · FAISS · sentence-transformers · Pydantic v2

---

## Dataset

Amazon Reviews 2023 (McAuley Lab, UC San Diego) , 48M items across 33 categories, ingested via HuggingFace `datasets`. Default config: `raw_meta_All_Beauty`, first 1,000 items for local development.
