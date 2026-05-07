from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Best model per provider — used for all agents.
# Override with LLM_MODEL in .env if needed.
#
# OpenAI:    openai:o4-mini  openai:o3  openai:gpt-4.1
# Google:    google-gla:gemini-2.5-pro  google-gla:gemini-2.5-flash
# Anthropic: anthropic:claude-sonnet-4-6  anthropic:claude-opus-4-6
_PROVIDER_MODEL: dict[str, str] = {
    "anthropic": "anthropic:claude-sonnet-4-6",
    "openai":    "openai:gpt-5.2",
    "google":    "google-gla:gemini-2.5-pro",
}


class Settings(BaseSettings):
    bee_api_key: str
    bee_api_base: str = "https://api.getbee.io"

    # Beefree SDK editor credentials (for embedding the visual editor)
    bee_client_id: str = ""
    bee_client_secret: str = ""

    # Set AI_PROVIDER in .env to switch between anthropic / openai / google.
    ai_provider: Literal["anthropic", "openai", "google"] = "anthropic"

    llm_model: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def resolve_model(self) -> "Settings":
        if not self.llm_model:
            self.llm_model = _PROVIDER_MODEL.get(self.ai_provider, _PROVIDER_MODEL["anthropic"])
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
