"""
Ingestion pipeline , one-shot ETL that:
  1. Loads the HuggingFace Amazon Reviews 2023 dataset
  2. Preprocesses each record (build embedding_text, normalise fields)
  3. Batch-encodes embedding_text with BAAI/bge-small-en-v1.5
  4. Bulk-indexes documents into OpenSearch via helpers.bulk()

Run directly:
    python -m ingestion.pipeline
or via Docker Compose (profile: ingest):
    docker compose --profile ingest up ingestion
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Generator, Iterator

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root → finds shared/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # services/ → finds api, ingestion, mlops

from datasets import load_dataset
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

from shared.config.settings import get_settings
from ingestion.schema import (
    HYBRID_PIPELINE_ID,
    HYBRID_SEARCH_PIPELINE,
    INDEX_SETTINGS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _to_list(value) -> list:
    """Normalize numpy arrays, lists, or None to a plain Python list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):   # catches numpy.ndarray
        return value.tolist()
    if isinstance(value, list):
        return value
    return [value]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_str(value: Any) -> str:
    """Convert any value to a non-empty string, empty string if falsy."""
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)


def _safe_float(value: Any) -> float | None:
    """Coerce to float; return None on failure."""
    if value is None:
        return None
    try:
        result = float(str(value).replace("$", "").replace(",", "").strip())
        return result if result >= 0 else None
    except (ValueError, AttributeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Coerce to int; return None on failure."""
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, AttributeError):
        return None


def _extract_image_url(images: Any) -> str:
    """Pull the first large image URL from the images field."""
    if not images:
        return ""
    if isinstance(images, list) and images:
        img = images[0]
        if isinstance(img, dict):
            return img.get("large", img.get("hi_res", img.get("thumb", ""))) or ""
    return ""




def content_hash(doc: dict) -> str:
    """SHA-256 of document content, excluding _id and _content_hash itself."""
    stable = {k: v for k, v in doc.items() if k not in ("_id", "_content_hash")}
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode()
    ).hexdigest()

def preprocess_record(record: dict) -> dict | None:
    """
    Transform a raw HuggingFace record into an OpenSearch document.

    Returns None for records that should be skipped (e.g. no ASIN).
    """
    parent_asin = record.get("parent_asin") or record.get("asin")
    if not parent_asin:
        return None

    title = _safe_str(record.get("title"))
    features_raw = _to_list(record.get("features"))
    features_text = _safe_str(features_raw)
    description_raw = _to_list(record.get("description"))
    description_text = _safe_str(
        description_raw[0] if description_raw
        else description_raw
    )

    # Build embedding_text: title + features + first description paragraph
    embedding_text = " ".join(filter(None, [title, features_text, description_text]))

    # Flatten details dict into top-level keys (store raw for reference)
    details = record.get("details") or {}
    if isinstance(details, str):
        details = {}
    
    doc = {
        "_id": parent_asin,
        "parent_asin": parent_asin,
        "title": title,
        "description": description_text,
        "features": features_text,
        "main_category": _safe_str(record.get("main_category")),
        "sub_category": _safe_str(record.get("sub_category")),
        "store": _safe_str(record.get("store")),
        "price": _safe_float(record.get("price")),
        "average_rating": _safe_float(record.get("average_rating")),
        "rating_number": _safe_int(record.get("rating_number")),
        "primary_image_url": _extract_image_url(record.get("images")),
        "embedding_text": embedding_text,
    }
    doc["_content_hash"] = content_hash(doc)  # ← hash computed after dict is built
    return doc


def iter_batches(
    records: list[dict], batch_size: int
) -> Generator[list[dict], None, None]:
    """Yield successive fixed-size batches from a list."""
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def build_bulk_actions(
    docs: list[dict], vectors: list[list[float]], index_name: str, client: OpenSearch
) -> Iterator[dict]:
    """Zip preprocessed documents with their embedding vectors, skipping unchanged docs."""
    # Fetch existing hashes for this batch in one mget round-trip
    ids = [doc["_id"] for doc in docs]
    existing = client.mget(index=index_name, body={"ids": ids})
    hash_map = {
        hit["_id"]: hit["_source"].get("_content_hash")
        for hit in existing["docs"]
        if hit.get("found")
    }

    skipped = 0
    for doc, vector in zip(docs, vectors):
        if hash_map.get(doc["_id"]) == doc["_content_hash"]:
            skipped += 1
            #print("identical data, skip reindex + re-embedding")
            continue  # ← identical data, skip reindex + re-embedding
        action = {
            "_index": index_name,
            "_id": doc["_id"],
            "_source": {k: v for k, v in doc.items() if k != "_id"},
        }
        action["_source"]["embedding_vector"] = vector
        yield action

    if skipped:
        logger.info("Skipped %d unchanged documents in this batch.", skipped)


# ── OpenSearch setup ──────────────────────────────────────────────────────────

def get_client(settings) -> OpenSearch:
    """Construct a synchronous OpenSearch client from settings."""
    http_auth = None
    if settings.opensearch_user and settings.opensearch_password:
        http_auth = (settings.opensearch_user, settings.opensearch_password)

    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=http_auth,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=False,
        timeout=60,
        retry_on_timeout=True,
        max_retries=3,
    )


def ensure_index(client: OpenSearch, index_name: str) -> None:
    """Create the index if it does not already exist."""
    if client.indices.exists(index=index_name):
        logger.info("Index '%s' already exists , skipping creation.", index_name)
        return
    logger.info("Creating index '%s'...", index_name)
    client.indices.create(index=index_name, body=INDEX_SETTINGS)
    logger.info("Index '%s' created successfully.", index_name)


def ensure_search_pipeline(client: OpenSearch) -> None:
    """Register the hybrid search pipeline if absent."""
    # NEW , opensearch-py 3.x compatible
    try:
        client.http.get(f"/_search/pipeline/{HYBRID_PIPELINE_ID}")
    except Exception:
        client.http.put(
            f"/_search/pipeline/{HYBRID_PIPELINE_ID}",
            body=HYBRID_SEARCH_PIPELINE,
        )


def wait_for_opensearch(client: OpenSearch, retries: int = 20, delay: float = 5.0) -> None:
    """Block until OpenSearch reports yellow or green cluster health."""
    logger.info("Waiting for OpenSearch to be ready...")
    for attempt in range(1, retries + 1):
        try:
            health = client.cluster.health(request_timeout=10)  # no timeout param
            status = health.get("status", "unknown")
            if status in ("yellow", "green"):
                logger.info("OpenSearch is ready (status: %s).", status)
                return
            logger.warning("Cluster status is '%s', retrying...", status)
        except Exception as exc:
            logger.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError("OpenSearch did not become ready in time.")

# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    settings = get_settings()

    logger.info("=== Product Ingestion Pipeline ===")
    logger.info("Dataset : %s / %s", settings.hf_dataset, settings.hf_dataset_config)
    logger.info("Index   : %s", settings.index_name)
    logger.info("Model   : %s", settings.embedding_model)

    # 1. Connect & wait for OpenSearch
    client = get_client(settings)
    wait_for_opensearch(client)
    ensure_index(client, settings.index_name)
    ensure_search_pipeline(client)

    # 2. Load model
    logger.info("Loading embedding model '%s'...", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Model loaded.")

    # 3. Load dataset
    logger.info("Loading HuggingFace dataset (this may take a while)...")
    dataset = load_dataset(
        settings.hf_dataset,
        settings.hf_dataset_config,
        split="full[:1000]",
        trust_remote_code=True,
    )
    logger.info("Dataset loaded , %d records.", len(dataset))

    # 4. Preprocess all records
    logger.info("Preprocessing records...")
    docs: list[dict] = []
    skipped = 0
    for raw in dataset:
        doc = preprocess_record(dict(raw))
        if doc is None:
            skipped += 1
            continue
        docs.append(doc)
    logger.info(
        "Preprocessing complete: %d documents, %d skipped.", len(docs), skipped
    )

    # 5. Encode + bulk-index in chunks
    total_indexed = 0
    total_errors = 0
    embedding_texts = [d["embedding_text"] for d in docs]

    logger.info(
        "Encoding and indexing in batches of %d (chunk_size=%d)...",
        settings.embedding_batch_size,
        settings.chunk_size,
    )

    # Process in embedding batches, then accumulate for bulk chunks
    accumulated_docs: list[dict] = []
    accumulated_vectors: list[list[float]] = []

    for batch_start in range(0, len(docs), settings.embedding_batch_size):
        batch_docs = docs[batch_start : batch_start + settings.embedding_batch_size]
        batch_texts = embedding_texts[batch_start : batch_start + settings.embedding_batch_size]

        vectors = model.encode(
            batch_texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        accumulated_docs.extend(batch_docs)
        accumulated_vectors.extend(vectors)

        # Flush to OpenSearch when we have enough for a bulk chunk
        while len(accumulated_docs) >= settings.chunk_size:
            chunk_docs = accumulated_docs[: settings.chunk_size]
            chunk_vecs = accumulated_vectors[: settings.chunk_size]
            accumulated_docs = accumulated_docs[settings.chunk_size :]
            accumulated_vectors = accumulated_vectors[settings.chunk_size :]

            actions = list(
                    build_bulk_actions(chunk_docs, chunk_vecs, settings.index_name, client)
                )
            success, errors = helpers.bulk(
                client,
                actions,
                chunk_size=settings.chunk_size,
                raise_on_error=False,
                request_timeout=120,
            )
            total_indexed += success
            total_errors += len(errors) if isinstance(errors, list) else 0
            logger.info(
                "Bulk chunk: %d indexed, %d errors | total indexed so far: %d",
                success,
                len(errors) if isinstance(errors, list) else 0,
                total_indexed,
            )

    # Flush remaining
    if accumulated_docs:
        actions = list(
            build_bulk_actions(accumulated_docs, accumulated_vectors, settings.index_name, client)
        )
        success, errors = helpers.bulk(
            client,
            actions,
            chunk_size=settings.chunk_size,
            raise_on_error=False,
            request_timeout=120,
        )
        total_indexed += success
        total_errors += len(errors) if isinstance(errors, list) else 0
        logger.info(
            "Final bulk flush: %d indexed, %d errors.", success,
            len(errors) if isinstance(errors, list) else 0,
        )

    logger.info(
        "=== Ingestion complete: %d documents indexed, %d errors ===",
        total_indexed,
        total_errors,
    )


if __name__ == "__main__":
    run_pipeline()
