"""
Centralised configuration for BotTrade.

All values are loaded from environment variables (via `.env`).
Access config everywhere with:

    from config.settings import settings
    print(settings.llm_model)
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Anthropic / Claude
    # ------------------------------------------------------------------ #
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    llm_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Anthropic model ID used by AITradingEngine",
    )
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.0)

    # ------------------------------------------------------------------ #
    # Boardroom: model per seat
    # ------------------------------------------------------------------ #
    # The boardroom fires ~9 calls per cycle: 8 analysts plus the chairman.
    # Running all nine on one model means any upgrade multiplies by nine.
    # Splitting them lets the chairman - the only seat whose output is
    # binding - be as strong as you like while the analysts, who each
    # produce one structured vote from a narrow packet, stay cheap.
    #
    # Both fall back to llm_model when unset, so a single-model setup keeps
    # working exactly as before.
    llm_model_analyst: Optional[str] = Field(
        default=None,
        description="Model for the 8 boardroom analysts. Defaults to LLM_MODEL.",
    )
    llm_model_chair: Optional[str] = Field(
        default=None,
        description="Model for the boardroom chairman, who casts the binding "
                    "ruling. Defaults to LLM_MODEL.",
    )

    # ------------------------------------------------------------------ #
    # Embeddings (local sentence-transformers)
    # ------------------------------------------------------------------ #
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model name",
    )

    # ------------------------------------------------------------------ #
    # ChromaDB
    # ------------------------------------------------------------------ #
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="Directory where ChromaDB persists its index",
    )
    chroma_collection_name: str = Field(
        default="trading_strategies",
        description="ChromaDB collection that stores all ingested knowledge",
    )

    # ------------------------------------------------------------------ #
    # Text chunking
    # ------------------------------------------------------------------ #
    chunk_size: int = Field(
        default=1000,
        description="Target character count per text chunk",
    )
    chunk_overlap: int = Field(
        default=150,
        description="Overlap between consecutive chunks (preserves context)",
    )

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------ #
    # Deployment environment
    # ------------------------------------------------------------------ #
    bottrade_env: str = Field(
        default="dev",
        description="Deployment environment: dev | staging | production. "
                    "Production turns on stricter defaults (debug off, "
                    "external notifications encouraged, no example data).",
    )
    bottrade_auth_password_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the dashboard password. When unset, "
                    "auth is disabled (local dev mode). Generate with "
                    "`python -c \"import hashlib;print(hashlib.sha256("
                    "b'mypassword').hexdigest())\"`.",
    )
    bottrade_auth_hash_salt: Optional[str] = Field(
        default=None,
        description="Optional salt prepended to the password before hashing.",
    )
    bottrade_port: int = Field(
        default=8501,
        description="Streamlit server port (matches docker-compose mapping).",
    )

    @property
    def is_production(self) -> bool:
        return self.bottrade_env.lower() == "production"

    @property
    def is_dev(self) -> bool:
        return self.bottrade_env.lower() in ("dev", "development", "local")


# Singleton — import this object everywhere
settings = Settings()
