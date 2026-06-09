"""
OpenSearch index mapping and settings for the products index.

Index design decisions:
- knn=true at index level enables approximate nearest neighbour queries
- faiss engine with innerproduct space (valid because embeddings are L2-normalised)
- hnsw parameters: ef_construction=128, m=16 → good recall/size trade-off
- title is mapped as both text (BM25) and keyword (exact filter / sort)
- Custom analysers: product_analyzer for full-text, autocomplete_analyzer for suggest
"""

from __future__ import annotations

INDEX_SETTINGS: dict = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": 3,
            "number_of_replicas": 1,
        },
        "analysis": {
            "tokenizer": {
                "edge_ngram_tokenizer": {
                    "type": "edge_ngram",
                    "min_gram": 2,
                    "max_gram": 20,
                    "token_chars": ["letter", "digit"],
                }
            },
            "analyzer": {
                "product_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                },
                "autocomplete_analyzer": {
                    "type": "custom",
                    "tokenizer": "edge_ngram_tokenizer",
                    "filter": ["lowercase"],
                },
                "autocomplete_search_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            # ── Textual fields ──────────────────────────────────────
            "title": {
                "type": "text",
                "analyzer": "product_analyzer",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 512},
                    "suggest": {
                        "type": "text",
                        "analyzer": "autocomplete_analyzer",
                        "search_analyzer": "autocomplete_search_analyzer",
                    },
                },
            },
            "description": {
                "type": "text",
                "analyzer": "product_analyzer",
            },
            "features": {
                "type": "text",
                "analyzer": "product_analyzer",
            },
            # ── Vector field ────────────────────────────────────────
            "embedding_vector": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "innerproduct",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16,
                    },
                },
            },
            # ── Categorical / filter fields ─────────────────────────
            "main_category": {"type": "keyword"},
            "sub_category": {"type": "keyword"},
            "parent_asin": {"type": "keyword"},
            "store": {"type": "keyword"},
            # ── Numeric fields ──────────────────────────────────────
            "price": {"type": "float"},
            "average_rating": {"type": "float"},
            "rating_number": {"type": "integer"},
            # ── Display-only fields (not indexed) ────────────────────
            "primary_image_url": {"type": "keyword", "index": False},
            "thumbnail_url": {"type": "keyword", "index": False},
            # ── Embedding source text (stored for debug) ─────────────
            "embedding_text": {
                "type": "text",
                "index": False,
                "store": True,
            },
        }
    },
}


# Search pipeline for hybrid normalization (registered once at startup)
HYBRID_PIPELINE_ID = "hybrid-search-pipeline"

HYBRID_SEARCH_PIPELINE: dict = {
    "description": "Normalise and merge BM25 + kNN scores",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {"weights": [0.4, 0.6]},
                },
            }
        }
    ],
}
