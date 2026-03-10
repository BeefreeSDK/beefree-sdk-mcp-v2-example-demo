from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bee_api_key: str
    bee_api_base: str = "https://api.getbee.io"
    llm_model: str = "anthropic:claude-sonnet-4-6"
    llm_planner_model: str = "anthropic:claude-haiku-4-5-20251001"
    llm_executor_model: str = "anthropic:claude-haiku-4-5-20251001"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
