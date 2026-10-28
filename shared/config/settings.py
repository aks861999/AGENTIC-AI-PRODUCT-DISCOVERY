"""
Shared settings module loaded via pydantic-settings.
All environment variables are declared here with typed defaults.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenSearch ────────────────────────────────────────────────
    opensearch_host: str = Field(default="localhost", alias="OPENSEARCH_HOST")
    opensearch_port: int = Field(default=9200, alias="OPENSEARCH_PORT")
    opensearch_use_ssl: bool = Field(default=False, alias="OPENSEARCH_USE_SSL")
    opensearch_verify_certs: bool = Field(default=False, alias="OPENSEARCH_VERIFY_CERTS")
    opensearch_user: Optional[str] = Field(default=None, alias="OPENSEARCH_USER")
    opensearch_password: Optional[str] = Field(default=None, alias="OPENSEARCH_PASSWORD")

    max_retries: int = Field(default=0, alias="MAX_RETRIES")

    # ── Index ─────────────────────────────────────────────────────
    index_name: str = Field(default="products", alias="INDEX_NAME")

    # ── Embedding model ───────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(default=128, alias="BATCH_SIZE")

    # ── Ingestion ─────────────────────────────────────────────────
    home_depot_desc_path: str = ""
    home_depot_attr_path: str = ""
    
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")

    # ── LLM / Google Generative AI ────────────────────────────────
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-2.0-flash", alias="LLM_MODEL")

    # Groq (fallback — used when Gemini quota is exhausted or key missing)
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")


    # ── API ───────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── Search defaults ───────────────────────────────────────────
    default_size: int = Field(default=10, alias="DEFAULT_SIZE")
    knn_k: int = Field(default=500, alias="KNN_K")
    # Written into the hybrid search pipeline (scaled to sum to 1) at ingestion and API startup
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
