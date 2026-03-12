from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default models per provider
# "main"  → used for layout agent, executor agents, single-email agent
# "fast"  → used for planner and translation agent (lower latency / cost)
#
# You can override any individual model in .env without changing AI_PROVIDER:
#   LLM_EXECUTOR_MODEL=openai:o3
#   LLM_LAYOUT_MODEL=openai:o4-mini
#   LLM_PLANNER_MODEL=openai:gpt-4o-mini
#
# OpenAI reasoning models:    openai:o4-mini  openai:o3  openai:o1
# Google thinking models:     google-gla:gemini-2.5-pro  google-gla:gemini-2.5-flash
# Anthropic:                  anthropic:claude-sonnet-4-6  anthropic:claude-opus-4-5
_PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {
        "main": "anthropic:claude-sonnet-4-6",
        "fast": "anthropic:claude-haiku-4-5-20251001",
    },
    "openai": {
        # o4-mini is OpenAI's latest compact reasoning model — significantly
        # better than gpt-4o at structured multi-step tool use tasks like this.
        "main": "openai:o4-mini",
        "fast": "openai:gpt-4o-mini",
    },
    "google": {
        # gemini-2.5-pro has thinking enabled by default and outperforms flash
        # on complex layout/content generation.
        "main": "google-gla:gemini-2.5-pro",
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
