"""
Shared settings module loaded via pydantic-settings.
All environment variables are declared here with typed defaults.
"""

from functools import lru_cache
from typing import Optional, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Agent mode ────────────────────────────────────────────────────────────
    agent_mode: Literal["search_only", "full_commerce"] = Field(
        default="full_commerce", alias="AGENT_MODE"
    )

    # ── OpenSearch ────────────────────────────────────────────────
    opensearch_host: str = Field(default="localhost", alias="OPENSEARCH_HOST")
    opensearch_port: int = Field(default=9200, alias="OPENSEARCH_PORT")
    opensearch_use_ssl: bool = Field(default=False, alias="OPENSEARCH_USE_SSL")
    opensearch_verify_certs: bool = Field(default=False, alias="OPENSEARCH_VERIFY_CERTS")
    opensearch_user: Optional[str] = Field(default=None, alias="OPENSEARCH_USER")
    opensearch_password: Optional[str] = Field(default=None, alias="OPENSEARCH_PASSWORD")

    # ── Index ─────────────────────────────────────────────────────
    index_name: str = Field(default="products", alias="INDEX_NAME")

    # ── Embedding model ───────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    embedding_batch_size: int = Field(default=128, alias="BATCH_SIZE")

    # ── Ingestion ─────────────────────────────────────────────────
    hf_dataset: str = Field(
        default="McAuley-Lab/Amazon-Reviews-2023", alias="HF_DATASET"
    )
    hf_dataset_config: str = Field(
        default="raw_meta_All_Beauty", alias="HF_DATASET_CONFIG"
    )
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")

    # ── LLM / Google Generative AI ────────────────────────────────
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-2.0-flash", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # ── API ───────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_workers: int = Field(default=1, alias="API_WORKERS")

    # ── Search defaults ───────────────────────────────────────────
    default_size: int = Field(default=10, alias="DEFAULT_SIZE")
    knn_k: int = Field(default=100, alias="KNN_K")
    hybrid_bm25_weight: float = Field(default=0.4, alias="HYBRID_BM25_WEIGHT")
    hybrid_knn_weight: float = Field(default=0.6, alias="HYBRID_KNN_WEIGHT")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @property
    def opensearch_url(self) -> str:
        scheme = "https" if self.opensearch_use_ssl else "http"
        return f"{scheme}://{self.opensearch_host}:{self.opensearch_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
