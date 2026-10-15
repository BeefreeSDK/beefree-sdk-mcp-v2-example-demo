from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default model per provider — used for all agents.
# Override with LLM_MODEL in .env if needed.
_PROVIDER_MODEL: dict[str, str] = {
    "anthropic": "anthropic:claude-sonnet-5-5",
    "openai": "openai:gpt-6.1-sol",
    "google": "google-gla:gemini-3.8-flash",
}

PROVIDER_MODELS = {
    "anthropic": ("Anthropic", [
        "anthropic:claude-sonnet-5-5", "anthropic:claude-opus-5-5",
        "anthropic:claude-haiku-5-5", "anthropic:claude-fable-5-1",
        "anthropic:claude-sonnet-4-6", "anthropic:claude-opus-4-6",
    ]),
    "openai": ("OpenAI", [
        "openai:gpt-6.1-sol", "openai:gpt-6-astra", "openai:gpt-6-luna",
        "openai:gpt-5.2", "openai:gpt-5-mini", "openai:gpt-4.1",
    ]),
    "google": ("Google", [
        "google-gla:gemini-3.8-flash", "google-gla:gemini-4-argon",
        "google-gla:gemini-3.5-flash-lite",
        "google-gla:gemini-2.5-pro", "google-gla:gemini-2.5-flash",
    ]),
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
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    google_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def agent_model(self) -> str:
        if self.llm_model.startswith("openai:"):
            return self.llm_model.replace("openai:", "openai-responses:", 1)
        return self.llm_model

    @model_validator(mode="after")
    def resolve_model(self) -> "Settings":
        if not self.llm_model:
            self.llm_model = _PROVIDER_MODEL.get(self.ai_provider, _PROVIDER_MODEL["anthropic"])
        return self

    def model_catalog(self) -> dict:
        providers = []
        for provider, (label, supported_models) in PROVIDER_MODELS.items():
            key = getattr(self, f"{provider}_api_key").get_secret_value().strip()
            if provider == "google" and not key:
                key = self.gemini_api_key.get_secret_value().strip()
            if not key:
                continue
            models = list(supported_models)
            prefix = "google-gla:" if provider == "google" else f"{provider}:"
            if self.llm_model.startswith(prefix) and self.llm_model not in models:
                models.insert(0, self.llm_model)
            providers.append({"id": provider, "label": label, "models": models})
        available = {model for provider in providers for model in provider["models"]}
        default = self.llm_model if self.llm_model in available else None
        return {"providers": providers, "default_model": default}

    def for_model(self, model: str | None = None) -> "Settings":
        catalog = self.model_catalog()
        selected = model if model is not None else catalog["default_model"]
        for provider in catalog["providers"]:
            if selected in provider["models"]:
                return self.model_copy(update={"ai_provider": provider["id"], "llm_model": selected})
        if selected is None:
            raise ValueError("Configure an API key and select an available model.")
        raise ValueError("Model unavailable: select a supported model with a configured API key.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
