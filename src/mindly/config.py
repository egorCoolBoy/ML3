from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-20b:free"
    llm_fallback_model: str = "qwen/qwen3-next-80b-a3b-instruct:free"
    embedding_model: str = "intfloat/multilingual-e5-small"
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    sqlite_path: Path = PROJECT_ROOT / "data" / "memory.db"
    log_file: Path = PROJECT_ROOT / "data" / "log_file.log"
    wandb_project: str = "mindly-memory"
    wandb_mode: str = "offline"
    retrieval_top_k: int = 5
    recent_turns_limit: int = 6
    proactive_word_threshold: int = 35
    max_input_tokens: int = 8000


def get_settings() -> Settings:
    return Settings()
