"""Markup checks for the Email sequence page and the live results partials."""
import re
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import main as routes
from app.plans import build_preset_plan

ROOT = Path(__file__).resolve().parent.parent
PALETTE = {"id": "ocean-blue", "name": "Ocean blue", "swatches": ["#000", "#fff"]}


def render(name, **context):
    return routes.templates.env.get_template(name).render(**context)


class ResultsPartialTests(unittest.TestCase):
    def partials(self):
        plan = build_preset_plan(
            "black-friday-fashion", "Test", routes.SEQUENCE_PRESETS["black-friday-fashion"], 0,
        )
        return {
            "sequence": render(
                "partials/sequence.html", plan=plan, llm_model="openai:gpt-5-mini",
                emails_with_ids=[(e, f"tpl{i}") for i, e in enumerate(plan.emails)],
        prompt_inputs={f"tpl{i}": f"in{i}" for i in range(len(plan.emails))},
            ),
            "palette": render(
                "partials/palette_sequence.html", llm_model="openai:gpt-5-mini",
                palettes_with_ids=[(PALETTE, "tplA"), (PALETTE, "tplB")],
            ),
            "translate": render(
                "partials/translate_sequence.html", llm_model="openai:gpt-5-mini",
                languages_with_ids=[("Spanish", "tplA"), ("German", "tplB")],
            ),
        }

    def test_shared_stylesheet_replaces_inline_styles(self):
        for name, html in self.partials().items():
            with self.subTest(partial=name):
                self.assertIn('<link rel="stylesheet" href="/static/results.css">', html)
                self.assertNotIn("<style>", html)
                self.assertNotIn("@keyframes", html)
        css = (ROOT / "static" / "results.css").read_text()
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn("minmax(min(280px, 100%), 1fr)", css)

    def test_outcomes_and_errors_handled_without_reconnecting(self):
        for name, html in self.partials().items():
            with self.subTest(partial=name):
                self.assertIn("addEventListener('outcome'", html)
                self.assertIn("htmx:sseError", html)
                self.assertIn("removeAttribute('sse-connect')", html)
                self.assertIn("Needs review", html)
                self.assertNotIn("alert(", html)
                self.assertNotIn("agents in parallel", html)
                self.assertNotIn("✓ Done", html)

    def test_download_buttons_use_shared_style(self):
        for name, html in self.partials().items():
            with self.subTest(partial=name):
                self.assertIn('class="btn-download"', html)
                self.assertNotIn("font-size:.72rem;padding:.22rem .55rem", html)

    def test_sequence_offers_partial_downloads(self):
        html = self.partials()["sequence"]
        self.assertIn("'Download ' + usable.length + ' of ' + total + ' emails'", html)
        self.assertIn('data-template-id="tpl2"', html)

    def test_loading_partials_stop_on_stream_errors(self):
        for name, context in [
            ("partials/palette_loading.html", {"session_id": "s", "palettes": [PALETTE]}),
            ("partials/translate_loading.html", {"session_id": "s", "languages": ["Spanish"]}),
        ]:
            with self.subTest(partial=name):
                html = render(name, **context)
                self.assertIn("htmx:sseError", html)
                self.assertIn("inline-error", html)


class SequencePageTests(unittest.TestCase):
    def test_page_structure(self):
        html = TestClient(routes.app).get("/sequence").text
        self.assertIn("<title>Beefree SDK MCP — Email sequence</title>", html)
        self.assertEqual(len(re.findall(r"<h1[\s>]", html)), 1)
        self.assertIn('href="/static/results.css"', html)
        self.assertIn('class="demo-disclaimer"', html)
        self.assertLess(html.index('class="demo-disclaimer"'), html.index('id="workspace"'))
        self.assertIn('class="brief-summary"', html)
        self.assertIn('hx-disabled-elt=', html)
        self.assertEqual(html.count('aria-pressed="false"'), 3)
        self.assertNotIn("rescaleThumbs", html)
        self.assertNotIn("getElementById('workspace').hidden", html)
        self.assertNotIn("alert(", html)
        if (ROOT / "static" / "vendor" / "htmx.min.js").exists():
            self.assertNotIn("unpkg.com", html)


if __name__ == "__main__":
    unittest.main()
