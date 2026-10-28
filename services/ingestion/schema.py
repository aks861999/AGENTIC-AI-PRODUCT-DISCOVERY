"""
OpenSearch index mapping and settings for the products index.

Index design decisions:
- knn=true at index level enables approximate nearest neighbour queries
- faiss engine with innerproduct space (valid because embeddings are L2-normalised)
- hnsw parameters: ef_construction=64, m=8 → small graph for a small instance
- title is mapped as both text (BM25) and keyword (exact filter / sort)
- Custom analysers: product_analyzer for full-text, autocomplete_analyzer for suggest
- embedding_vector dimension is set from the embedding model (build_index_settings)
"""
from __future__ import annotations

import copy

INDEX_SETTINGS: dict = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "mapping.total_fields.limit": 1000,
            "max_result_window": 10000,  
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
        "dynamic": "strict",
        "properties": {
            "product_uid": {"type": "keyword"},
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
            "product_description": {
                "type": "text",
                "analyzer": "product_analyzer",
            },
            "bullet_points": {
                "type": "text",
                "analyzer": "product_analyzer",
            },
            "attributes": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "value": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    }
                }
            },
            "embedding_vector": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "innerproduct",
                    "parameters": {
                        "ef_construction": 64,
                        "m": 8,
                    },
                },
            },
        }
    },
}

def build_index_settings(embedding_dim: int) -> dict:
    """INDEX_SETTINGS with the vector dimension matching the embedding model."""
    body = copy.deepcopy(INDEX_SETTINGS)
    body["mappings"]["properties"]["embedding_vector"]["dimension"] = embedding_dim
    return body


HYBRID_PIPELINE_ID = "hybrid-search-pipeline"


def build_hybrid_search_pipeline(bm25_weight: float, knn_weight: float) -> dict:
    """
    Normalise and merge BM25 + kNN scores. Weights follow the sub-query order
    in api.search._build_hybrid_query and are scaled to sum to 1.
    """
    total = bm25_weight + knn_weight
    return {
        "description": "Normalise and merge BM25 + kNN scores",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [bm25_weight / total, knn_weight / total]},
                    },
                }
            }
        ],
    }