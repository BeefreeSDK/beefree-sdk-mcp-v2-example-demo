import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse

from app import main
from app.config import Settings
from app.payload import chat_session, chat_template, unpack


def settings(key="test"):
    return Settings(
        _env_file=None, bee_api_key=key, ai_provider="openai", llm_model="",
        anthropic_api_key="", google_api_key="", gemini_api_key="", openai_api_key="test-secret",
    )


class ChatSessionTests(unittest.TestCase):
    """Chats keep nothing in server memory, so each request can reach any app copy."""

    def setUp(self):
        patcher = patch.object(main, "get_settings", return_value=settings())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def run_turn(self, session_id, fields):
        captured = {}

        async def fake_executor(**kwargs):
            captured.update(kwargs)
            yield {"event": "close", "data": ""}

        token = self.client.post("/run-input", json=fields).json()["input_id"]
        with patch.object(main, "stream_edit_executor", fake_executor):
            body = self.client.get("/edit-stream", params={"session_id": session_id, "input_id": token}).text
        return captured, body

    def test_session_token_names_the_template_and_is_signed(self):
        token = chat_session("tmpl-1", "secret")
        self.assertEqual(chat_template(token, "secret"), "tmpl-1")
        self.assertIsNone(chat_template(token, "another-secret"))
        forged = main.pack({"t": "someone-elses-template", "s": "0" * 32})
        self.assertIsNone(chat_template(forged, "secret"))

    def test_integration_start_and_refresh(self):
        with patch.object(main, "create_template", AsyncMock(return_value="tmpl-1")):
            data = self.client.post("/integration-start").json()
        self.assertEqual(chat_template(data["session_id"], "test"), "tmpl-1")
        refreshed = self.client.post("/integration-refresh", params={
            "session_id": data["session_id"], "template_id": "tmpl-2"}).json()
        self.assertEqual(chat_template(refreshed["session_id"], "test"), "tmpl-2")
        bad = self.client.post("/integration-refresh", params={"session_id": "nope", "template_id": "x"})
        self.assertEqual(bad.status_code, 404)

    def test_turn_runs_on_the_template_with_the_page_history(self):
        session = chat_session("tmpl-1", "test")
        captured, _ = self.run_turn(session, {
            "message": "Make it bigger",
            "history": [["Add a red button", "Done. I added a red button under the headline."]],
        })
        self.assertEqual(captured["template_id"], "tmpl-1")
        history = captured["message_history"]
        self.assertEqual(len(history), 2)
        self.assertIsInstance(history[0], ModelRequest)
        self.assertEqual(history[0].parts[0].content, "Add a red button")
        self.assertIsInstance(history[1], ModelResponse)
        self.assertIn("red button", history[1].parts[0].content)

    def test_history_is_trimmed_and_malformed_turns_ignored(self):
        turns = [[f"ask {i}", "x" * 2000] for i in range(10)] + [["only one"], [1, 2]]
        token = self.client.post("/run-input", json={"message": "hi", "history": turns}).json()["input_id"]
        # The last 6 turns are kept; 2 of those are malformed and dropped.
        saved = unpack(token)["history"]
        self.assertEqual([user for user, _ in saved], ["ask 6", "ask 7", "ask 8", "ask 9"])
        self.assertTrue(all(len(agent) == main.CHAT_TURN_CHARS for _, agent in saved))
        self.assertEqual(len(main.chat_history(turns)), 8)   # 4 turns, a request and a reply each

    def test_unknown_session(self):
        captured, body = self.run_turn("not-a-session", {"message": "hi"})
        self.assertEqual(captured, {})
        self.assertIn("could not be found", body)


if __name__ == "__main__":
    unittest.main()


class PaletteCorrectionTests(unittest.TestCase):
    """A palette version finished by the safety net is saved as a new Beefree
    template, so its download works from any app copy (nothing kept in memory)."""

    def test_corrected_copy_becomes_the_download(self):
        fixed = {"page": {"rows": []}}

        async def fake_executor(template_id, palette, settings):
            yield {"event": "fixed-template", "data": fixed}
            yield {"event": "fix-note", "data": "Completed automatically."}
            yield {"event": "close", "data": ""}

        seed = AsyncMock(return_value="fixed-tmpl")
        with (
            patch.object(main, "get_settings", return_value=settings()),
            patch.object(main, "stream_palette_executor", fake_executor),
            patch.object(main, "create_seeded_template", seed),
        ):
            body = TestClient(main.app).get(f"/stream-palette/orig?palette_id={main.PALETTES[0]['id']}").text
        self.assertEqual(seed.call_args.args[1], fixed)
        self.assertIn("event: download-template\r\ndata: fixed-tmpl", body)
        self.assertNotIn("fixed-template", body)
        self.assertIn("event: fix-note", body)
