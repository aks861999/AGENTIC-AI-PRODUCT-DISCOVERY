# services/scripts/build_schema_cache.py
# Run once: python -m scripts.build_schema_cache
# Saves: services/api/schema_cache.npz

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import json
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
from shared.config.settings import get_settings

CACHE_PATH = Path(__file__).resolve().parents[1] / "api" / "schema_cache.npz"

async def build():
    settings = get_settings()
    http_auth = (settings.opensearch_user, settings.opensearch_password) if settings.opensearch_user else None
    client = OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=http_auth,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=False,
        timeout=60,
        connection_class=RequestsHttpConnection,
    )

    print("Fetching attribute names from OpenSearch...")
    raw = await asyncio.to_thread(
        client.search,
        index=settings.index_name,
        body={
            "size": 0,
            "aggs": {
                "attr_names": {
                    "nested": {"path": "attributes"},
                    "aggs": {
                        "names": {
                            "terms": {"field": "attributes.name", "size": 5000, "order": {"_count": "desc"}}
                        }
                    }
                }
            }
        },
        request_timeout=30,
    )

    buckets = raw.get("aggregations", {}).get("attr_names", {}).get("names", {}).get("buckets", [])
    names = [b["key"] for b in buckets if b.get("key")]

    if not names:
        print("ERROR: No attribute names returned. Check your index name and OpenSearch connection.")
        return

    print(f"Found {len(names)} unique attribute names. Sample: {names[:10]}")

    print("Embedding attribute names...")
    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(names, normalize_embeddings=True, show_progress_bar=True)

    np.savez_compressed(
        CACHE_PATH,
        names=np.array(names, dtype=object),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"\nSaved to: {CACHE_PATH}")
    print(f"  names shape   : ({len(names)},)")
    print(f"  embeddings shape: {embeddings.shape}")

    # Verify the target attribute is present
    target = "Color/Finish"
    if target in names:
        print(f"\n  '{target}' found at index {names.index(target)}  ← GOOD")
    else:
        print(f"\n  WARNING: '{target}' NOT in the attribute list — check your index!")

    # Print full list to confirm
    print("\nFull attribute name list:")
    for i, n in enumerate(names):
        print(f"  [{i:03d}] {n}")

if __name__ == "__main__":
    asyncio.run(build())