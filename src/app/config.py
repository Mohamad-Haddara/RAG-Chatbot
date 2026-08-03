"""
Parsing Environment Variables with Pydantic

Alongside the BaseModel, Pydantic also implements a Base class for parsing settings and secrets from files.

The BaseSettings class provides a Pydantic features for loading settings or config class from environment variables or secret files.

Using this feature, the settings values can be set in code or overridden by environment variables.

This is useful in production where I don't want to expose secrets inside the code or the container environment.

When I create a model inheriting from BaseSettings, the model initializer will attempt to set values of each field using provided
defaults.

12-factor config — the SAME code runs in dev / staging / prod;
   only environment variables change. So anything that differs per
   environment must become a field here.

Pydantic validates types and required values at startup, so bad or missing setting crashes immediately with clear error


aliases = env var names
"""

from pathlib import Path

# Use BaseSettings to parse environment variables
from typing import Annotated
from pydantic import Field, HttpUrl, PostgresDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

CONFIG_FILE = Path(__file__).resolve()
APP_DIR = CONFIG_FILE.parent
PROJECT_ROOT = APP_DIR.parent.parent
DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(exist_ok=True)


# AppSettings inherit from BaseSettings - 
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file= str(PROJECT_ROOT / ".env"),
        env_file_encoding= "utf-8", # Configure AppSettings to read environment variables from ENV file at the root of a project with the UTS-8 encoding,
        extra= "ignore",
        case_sensitive= False
    )

    # snake_case field will map to environment variables names that are an uppercase
    port: Annotated[int, Field(default=8000)]
    #app_secret: Annotated[str, Field(min_length=32)]

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    def token(self) -> str | None:
        return self.hf_token


    router_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        alias="ROUTER_MODEL",
    )

    generation_model: str = Field(
        default="google/gemma-4-31B-it",
        alias="GENERATION_MODEL",
    )

    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        alias="RERANKER_MODEL"
    )

    def get_default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_device: str = Field(default_factory=get_default_device, alias="RERANKER_DEVICE")

    # --- NER enrichment ---
    ner_enabled: bool = Field(default=True, alias="NER_ENABLED")
    ner_device: str = Field(default="cuda", alias="NER_DEVICE")
    ner_model_en: str = Field(default="dslim/bert-base-NER", alias="NER_MODEL_EN")
    ner_model_ar: str = Field(default="hatmimoha/arabic-ner", alias="NER_MODEL_AR")

    ner_batch_size: int = Field(default=32, alias="NER_BATCH_SIZE")
    query_transform_enabled: bool = Field(default=True, alias="QUERY_TRANSFORM_ENABLED")

    hybrid_candidate_multiplier: int = Field(
        default=2, alias="HYBRID_CANDIDATE_MULTIPLIER"
    )

    rag_min_relevance: float = Field(default=0.5, alias="RAG_MIN_RELEVANCE")

    jwt_secret_key: str = Field(
        default="change-me-in-production-use-a-real-secret", alias="JWT_SECRET_KEY"
    )

    jwt_expire_minutes: int = Field(
        default=1440, alias="JWT_EXPIRE_MINUTES"
    )

    db_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DB_DIR")
    database_url: str = Field(default="", alias="DATABASE_URL")

    @model_validator(mode="after")
    def set_database_url(self):
        if not self.database_url:
            self.database_url = f"sqlite+aiosqlite:///{self.db_dir}/app.db"
        return self

    local_data_path: Path = Field(
        default=PROJECT_ROOT / ".langchain_qdrant-2", alias="LOCAL_DATA_PATH"
    )
    qdrant_collection_name: str = Field(
        default="big_token_corpus", alias="QDRANT_COLLECTION_NAME"
    )




    # --- Self-Feedback Loop ---
    self_feedback_enabled: bool = Field(default=False, alias="SELF_FEEDBACK_ENABLED")
    self_feedback_max_iterations: int = Field(
        default=2, alias="SELF_FEEDBACK_MAX_ITERATIONS"
    )
    self_feedback_threshold: float = Field(default=0.7, alias="SELF_FEEDBACK_THRESHOLD")
    self_feedback_max_latency_ms: int = Field(
        default=15000, alias="SELF_FEEDBACK_MAX_LATENCY_MS"
    )


# We can check the AppSettings class is working by printing a dump of the model

settings = AppSettings()

if __name__ == "__main__":
    import json

    print(json.dumps(settings.model_dump(), indent=2, default=str))
