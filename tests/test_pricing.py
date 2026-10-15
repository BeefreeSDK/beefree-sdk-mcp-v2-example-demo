import json
import unittest
from types import SimpleNamespace

from app.agent import _tokens_event
from app.config import PROVIDER_MODELS
from app.pricing import PRICES, request_cost


class PricingTests(unittest.TestCase):
    def test_every_selectable_model_has_a_price(self):
        for _, models in PROVIDER_MODELS.values():
            for model in models:
                with self.subTest(model=model):
                    self.assertIn(model, PRICES)

    def test_uncached_input_and_output(self):
        cost = request_cost(
            "anthropic:claude-opus-5-5",
            input_tokens=1_000_000, output_tokens=1_000_000,
        )
        self.assertAlmostEqual(cost, 4.00 + 20.00)

    def test_cache_tokens_are_part_of_input_and_billed_at_cache_rates(self):
        # 1M prompt tokens: 600k cache reads, 300k cache writes, 100k uncached.
        cost = request_cost(
            "anthropic:claude-sonnet-4-6",
            input_tokens=1_000_000, output_tokens=0,
            cache_read_tokens=600_000, cache_write_tokens=300_000,
        )
        self.assertAlmostEqual(cost, 0.1 * 3.00 + 0.3 * 3.75 + 0.6 * 0.30)

    def test_long_prompt_switches_to_higher_tier(self):
        short = request_cost("anthropic:claude-haiku-5-5", input_tokens=100_000, output_tokens=0)
        long = request_cost("anthropic:claude-haiku-5-5", input_tokens=100_001, output_tokens=0)
        self.assertAlmostEqual(short, 0.1 * 0.10)
        self.assertAlmostEqual(long, 0.100001 * 0.50)

    def test_unknown_model_has_no_cost(self):
        self.assertIsNone(request_cost("openai:not-a-model", input_tokens=10, output_tokens=10))

    def test_tokens_event_carries_cost(self):
        usage = SimpleNamespace(
            input_tokens=2_000, output_tokens=500,
            cache_write_tokens=0, cache_read_tokens=1_000,
        )
        data = json.loads(_tokens_event(usage, "openai:gpt-6.1-sol")["data"])
        self.assertEqual(data["input"], 2_000)
        self.assertAlmostEqual(data["cost"], (1_000 * 2.00 + 1_000 * 0.10 + 500 * 10.00) / 1e6)
        unpriced = json.loads(_tokens_event(usage, "openai:custom")["data"])
        self.assertIsNone(unpriced["cost"])


if __name__ == "__main__":
    unittest.main()
