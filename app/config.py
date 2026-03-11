from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default models per provider
# "main"  → used for layout agent, executor agents, single-email agent
# "fast"  → used for planner and translation agent (lower latency / cost)
_PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {
        "main": "anthropic:claude-sonnet-4-6",
        "fast": "anthropic:claude-haiku-4-5-20251001",
    },
    "openai": {
        "main": "openai:gpt-4o",
        "fast": "openai:gpt-4o-mini",
    },
    "google": {
        "main": "google-gla:gemini-2.5-flash",
        "fast": "google-gla:gemini-2.5-flash",
    },
}


class Settings(BaseSettings):
    bee_api_key: str
    bee_api_base: str = "https://api.getbee.io"

    # Set AI_PROVIDER in .env to switch between anthropic / openai / google.
    # The four model fields below are auto-populated from the provider defaults
    # unless you explicitly override them in .env.
    ai_provider: Literal["anthropic", "openai", "google"] = "anthropic"

    llm_model: str = ""
    llm_planner_model: str = ""
    llm_layout_model: str = ""
    llm_executor_model: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def resolve_models(self) -> "Settings":
        defaults = _PROVIDER_MODELS.get(self.ai_provider, _PROVIDER_MODELS["anthropic"])
        if not self.llm_model:
            self.llm_model = defaults["main"]
        if not self.llm_planner_model:
            self.llm_planner_model = defaults["fast"]
        if not self.llm_layout_model:
            self.llm_layout_model = defaults["main"]
        if not self.llm_executor_model:
            self.llm_executor_model = defaults["main"]
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
