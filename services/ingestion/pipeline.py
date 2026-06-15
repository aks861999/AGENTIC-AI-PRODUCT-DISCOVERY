"""
Ingestion pipeline using Pandas for Home Depot catalog data.
Implements run-on text correction and comprehensive bulk-indexing error logging.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

def _safe_str(value: Any) -> str:
    if pd.isna(value) or not value:
        return ""
    return str(value).strip()

def content_hash(doc: dict) -> str:
    stable = {k: v for k, v in doc.items() if k not in ("_id", "_content_hash")}
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode()
    ).hexdigest()

def preprocess_record(row: dict) -> tuple[dict, str] | None:
    product_uid = _safe_str(row.get("product_uid"))
    if not product_uid:
        return None

    raw_description = _safe_str(row.get("product_description"))
    raw_attributes = row.get("raw_attributes", [])

    bullets = []
    clean_attributes = []

    for attr in raw_attributes:
        name = _safe_str(attr.get("name"))
        val = _safe_str(attr.get("value"))
        if not name or not val:
            continue
        if name.startswith("Bullet"):
            bullets.append(val)  # just collect values, not (name, val) tuples
        else:
            clean_attributes.append({"name": name, "value": val})

    # Sort by original BulletNN order to preserve sequence
    bullets.sort()  # Bullet01, Bullet02, ...

    # Dedupe: remove bullets that already appear in the description
    unique_bullets = []
    desc_lower = raw_description.lower()
    for b in bullets:
        if b.lower() not in desc_lower:
            unique_bullets.append(b)

    # Build clean description: original desc + any bullets not already present
    if unique_bullets:
        bullet_text = " ".join(unique_bullets)
        clean_description = f"{raw_description} {bullet_text}".strip()
    else:
        clean_description = raw_description

    clean_description = " ".join(clean_description.split())

    # Build title from first bullet or first 80 chars of description
    # (Home Depot has no explicit title field)
    if bullets:
        title = bullets[0][:120]
    else:
        title = clean_description[:120]

    # Build embedding text: description + bullet points + structured attributes
    attr_text = " ".join([f"{a['name']}: {a['value']}" for a in clean_attributes])
    embedding_text = " ".join(filter(None, [clean_description, " ".join(bullets), attr_text]))

    doc = {
        "_id": product_uid,
        "product_uid": product_uid,
        "title": title,
        "product_description": clean_description,
        "bullet_points": bullets,           # NEW: searchable bullet text array
        "attributes": clean_attributes,
    }
    return doc, embedding_text

def build_bulk_actions(
    docs: list[dict], vectors: list[list[float]], index_name: str, client: OpenSearch
) -> Iterator[dict]:
    """Build bulk actions with idempotent index (upsert semantics).
    
    NOTE: Content-hash deduplication removed for free-tier compatibility.
    On t3.small, mget() per batch adds ~100ms latency and memory pressure.
    For incremental updates, use index aliases and full reindex instead.
    """
    for doc, vector in zip(docs, vectors):
        # Use _id from doc; OpenSearch will update if exists
        action = {
            "_index": index_name,
            "_id": doc["_id"],
            "_source": {k: v for k, v in doc.items() if k != "_id" and k != "_content_hash"},
        }
        action["_source"]["embedding_vector"] = vector
        yield action

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
        connection_class="requests",
        pool_maxsize=10,
        pool_block=False,
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

def run_pipeline() -> None:
    settings = get_settings()

    logger.info("=== Product Ingestion Pipeline ===")
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

    # 3. Load structured source tables via Pandas
    logger.info("Parsing structured source tables...")
    desc_df = pd.read_csv(settings.home_depot_desc_path, encoding="ISO-8859-1")
    attr_df = pd.read_csv(settings.home_depot_attr_path, encoding="ISO-8859-1")

    # Group attributes into records list to map raw collections safely
    grouped_attrs = attr_df.groupby('product_uid').apply(
        lambda x: x[['name', 'value']].to_dict('records')
    ).reset_index(name='raw_attributes')

    merged_df = pd.merge(desc_df, grouped_attrs, on='product_uid', how='left')
    merged_df['raw_attributes'] = merged_df['raw_attributes'].apply(lambda x: x if isinstance(x, list) else [])

    # 4. Preprocess all records
    logger.info("Preprocessing records...")
    docs: list[dict] = []
    embedding_texts: list[str] = []
    skipped = 0
    
    for _, row in merged_df.iterrows():
        res = preprocess_record(row.to_dict())
        if res is None:
            skipped += 1
            continue
        docs.append(res[0])
        embedding_texts.append(res[1])
        
    logger.info(
        "Preprocessing complete: %d documents, %d skipped.", len(docs), skipped
    )

    # 5. Encode + bulk-index in chunks
    total_indexed = 0
    total_errors = 0

    logger.info(
        "Encoding and indexing in batches of %d (chunk_size=%d)...",
        settings.embedding_batch_size,
        settings.chunk_size,
    )

    # Temporarily set refresh interval to off to accelerate batch execution performance
    client.indices.put_settings(index=settings.index_name, body={"index": {"refresh_interval": "-1"}})
    logger.info("Refresh interval disabled for bulk indexing.")

    try:

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

        # Flush remaining records
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

    finally:
    # Restore standard search parameters
        client.indices.put_settings(index=settings.index_name, body={"index": {"refresh_interval": "1s"}})
        logger.info("Refresh interval restored to 1s.")
    client.indices.forcemerge(index=settings.index_name, max_num_segments=1)
    
    logger.info(
        "=== Ingestion complete: %d documents indexed, %d errors ===",
        total_indexed,
        total_errors,
    )

if __name__ == "__main__":
    run_pipeline()