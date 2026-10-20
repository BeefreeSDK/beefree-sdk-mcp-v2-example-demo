import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import main
from app.config import Settings


def settings():
    return Settings(
        _env_file=None, bee_api_key="test", ai_provider="openai", llm_model="",
        anthropic_api_key="", google_api_key="", gemini_api_key="", openai_api_key="test-secret",
    )


class RunInputTests(unittest.TestCase):
    """Long inputs travel in a POST; stream URLs only carry a short id (no 414s behind proxies)."""

    def setUp(self):
        patcher = patch.object(main, "get_settings", return_value=settings())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def test_sequence_stream_url_is_short(self):
        plan = self.client.post("/plan", data={"preset_id": "black-friday-fashion", "goal": "Test campaign"})
        self.assertEqual(plan.status_code, 200)
        plan_json = plan.text.split('name="plan_json" value="', 1)[1].split('"', 1)[0]
        import html
        loading = self.client.post("/execute", data={"plan_json": html.unescape(plan_json)}).text
        url = loading.split('sse-connect="', 1)[1].split('"', 1)[0]
        self.assertTrue(url.startswith("/execute-stream?input_id="))
        self.assertLess(len(url), 80)

    def test_single_email_stream_url_is_short(self):
        brief = "Goal: welcome email. " * 400   # ~8 KB, longer than common proxy limits
        partial = self.client.post("/generate", data={"brief": brief}).text
        self.assertNotIn("/generate-stream?brief=", partial)
        input_id = json.loads(partial.split("encodeURIComponent(", 1)[1].split(")", 1)[0])

        seen = {}

        async def fake_stream(template_id, got_brief, *args, **kwargs):
            seen["brief"] = got_brief
            yield {"event": "close", "data": ""}

        with (
            patch.object(main, "create_template", AsyncMock(return_value="tmpl")),
            patch.object(main, "stream_single_executor", fake_stream),
        ):
            self.client.get("/generate-stream", params={"input_id": input_id})
        self.assertEqual(seen["brief"], brief)

    def test_run_input_endpoint(self):
        response = self.client.post("/run-input", json={"message": "make it red"})
        input_id = response.json()["input_id"]
        self.assertEqual(main.run_inputs[input_id], {"message": "make it red"})
        self.assertEqual(self.client.post("/run-input", json={"plan_json": "{}"}).status_code, 400)
        self.assertEqual(self.client.post("/run-input", json={"brief": 5}).status_code, 400)
        self.assertEqual(self.client.post("/run-input", json={"brief": "x" * 70_000}).status_code, 413)

    def test_expired_input_explains_itself(self):
        body = self.client.get("/generate-stream", params={"input_id": "gone"}).text
        self.assertIn("no longer available", body)
        body = self.client.get("/codemode-stream", params={"input_id": "gone"}).text
        self.assertIn("no longer available", body)
        body = self.client.get("/execute-stream", params={"input_id": "gone"}).text
        self.assertIn("no longer available", body)
        main.edit_sessions["s1"] = {"template_id": "t", "messages": []}
        body = self.client.get("/edit-stream", params={"session_id": "s1", "input_id": "gone"}).text
        self.assertIn("no longer available", body)

    def test_pages_do_not_put_text_in_stream_urls(self):
        for path in ("/codemode", "/builder", "/integration"):
            html = self.client.get(path).text
            self.assertNotIn("'&message=' + encodeURIComponent", html, path)
            self.assertNotIn("-stream?brief=", html, path)


if __name__ == "__main__":
    unittest.main()
