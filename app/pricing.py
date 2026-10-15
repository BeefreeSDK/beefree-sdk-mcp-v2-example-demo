"""Per-model token prices, used to show what a generation cost.

USD per million tokens, standard tier, taken from the providers' pricing pages
on 2026-10-16:
- https://platform.claude.com/docs/en/about-claude/pricing
- https://developers.openai.com/api/docs/pricing
- https://ai.google.dev/gemini-api/docs/pricing

OpenAI and Google bill cache writes at the normal input price, so their
cache_write equals input. Update this table when prices change; a model that
is missing here shows token counts without a dollar amount.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    cache_write: float
    cache_read: float
    output: float


@dataclass(frozen=True)
class TieredPrice:
    """Price that switches to `above` once a request's prompt exceeds `threshold` tokens."""
    base: Price
    above: Price
    threshold: int


PRICES: dict[str, Price | TieredPrice] = {
    # Anthropic (cache_write is the 5-minute write rate)
    "anthropic:claude-fable-5-1": Price(10.00, 12.50, 0.25, 50.00),
    "anthropic:claude-opus-5-5": Price(4.00, 5.00, 0.20, 20.00),
    "anthropic:claude-opus-4-6": Price(5.00, 6.25, 0.50, 25.00),
    "anthropic:claude-sonnet-5-5": Price(2.00, 2.50, 0.10, 10.00),
    "anthropic:claude-sonnet-4-6": Price(3.00, 3.75, 0.30, 15.00),
    "anthropic:claude-haiku-5-5": TieredPrice(
        base=Price(0.10, 0.125, 0.01, 0.50),
        above=Price(0.50, 0.625, 0.05, 2.50),
        threshold=100_000,
    ),
    # OpenAI (long-context surcharges on gpt-6-astra / gpt-6-luna are not modelled)
    "openai:gpt-6.1-sol": Price(2.00, 2.00, 0.10, 10.00),
    "openai:gpt-6-astra": Price(10.00, 10.00, 1.00, 50.00),
    "openai:gpt-6-luna": Price(0.10, 0.10, 0.01, 0.50),
    "openai:gpt-5.2": Price(1.75, 1.75, 0.175, 14.00),
    "openai:gpt-5-mini": Price(0.25, 0.25, 0.025, 2.00),
    "openai:gpt-4.1": Price(2.00, 2.00, 0.50, 8.00),
    # Google. Gemini 4 Argon doubles on 2026-12-01 and Gemini 3.8 Flash on
    # 2027-01-01; these are the current rates.
    "google-gla:gemini-3.8-flash": Price(0.75, 0.75, 0.075, 3.75),
    "google-gla:gemini-4-argon": Price(2.00, 2.00, 0.10, 10.00),
    "google-gla:gemini-3.5-flash-lite": Price(0.30, 0.30, 0.03, 2.50),
    "google-gla:gemini-2.5-pro": TieredPrice(
        base=Price(1.25, 1.25, 0.125, 10.00),
        above=Price(2.50, 2.50, 0.25, 15.00),
        threshold=200_000,
    ),
    "google-gla:gemini-2.5-flash": Price(0.30, 0.30, 0.03, 2.50),
}


def request_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float | None:
    """Return the USD cost of one model request, or None if the model has no price.

    `input_tokens` is the total prompt size and already includes cache reads
    and writes (the PydanticAI / genai-prices convention).
    """
    price = PRICES.get(model)
    if price is None:
        return None
    if isinstance(price, TieredPrice):
        price = price.above if input_tokens > price.threshold else price.base
    uncached = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    return (
        uncached * price.input
        + cache_write_tokens * price.cache_write
        + cache_read_tokens * price.cache_read
        + output_tokens * price.output
    ) / 1_000_000
