import unittest

from fastapi.testclient import TestClient

from app.main import app


class BuilderPageTests(unittest.TestCase):
    def test_home_links_to_editor_and_headless_pages(self):
        response = TestClient(app).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('href="/editor"', response.text)
        self.assertIn('href="/headless"', response.text)
        self.assertNotIn('id="editor-modes"', response.text)
        self.assertNotIn('id="headless-modes"', response.text)

    def test_editor_page_has_two_cards_in_requested_order(self):
        response = TestClient(app).get("/editor")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="editor-modes"', response.text)
        self.assertNotIn('id="headless-modes"', response.text)
        self.assertLess(response.text.index('href="/builder"'), response.text.index('href="/integration"'))

    def test_headless_page_lists_headless_modes(self):
        response = TestClient(app).get("/headless")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="headless-modes"', response.text)
        self.assertNotIn('id="editor-modes"', response.text)
        for path in ["/single", "/palette", "/translate", "/edit", "/codemode", "/sequence"]:
            self.assertIn(f'href="{path}"', response.text)

    def test_simple_builder_reuses_editor_without_guide(self):
        response = TestClient(app).get("/builder")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="em-editor-container"', response.text)
        self.assertIn('id="em-input"', response.text)
        self.assertIn('window.initEditorManaged();', response.text)
        self.assertIn('window.startMcpSession();', response.text)
        self.assertNotIn('id="panel-api-managed"', response.text)
        self.assertNotIn('id="integration-guide"', response.text)
        self.assertNotIn('class="demo-tabs"', response.text)
        self.assertIn('src="/static/model-selection.js"', response.text)

    def test_guide_keeps_both_session_demos(self):
        response = TestClient(app).get("/integration")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="panel-api-managed"', response.text)
        self.assertIn('id="integration-guide"', response.text)
        self.assertNotIn('window.initEditorManaged();', response.text)


class IntegrationTemplateUxTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_builder_and_guide_share_accessibility_basics(self):
        for path in ("/builder", "/integration"):
            html = self.client.get(path).text
            self.assertEqual(html.count("<h1"), 1, path)
            self.assertNotIn("alert(", html, path)
            self.assertNotIn("beeTokensAdd", html, path)
            self.assertNotIn("height:760px", html, path)
            self.assertIn('aria-label="Send message"', html, path)
            self.assertIn('aria-label="Reset token counter"', html, path)
            self.assertIn('aria-expanded="true" aria-controls="em-chat-panel"', html, path)
            self.assertIn('class="inline-error" id="em-error" role="alert"', html, path)
            self.assertIn('<nav class="breadcrumb" aria-label="Breadcrumb">', html, path)
            self.assertIn('<li><a href="/editor">In the editor</a></li>', html, path)
            self.assertIn('class="logo-link"', html, path)
            self.assertNotIn('class="back-link"', html, path)
            self.assertIn('<script src="/static/ui.js"></script>', html, path)

    def test_builder_title_and_plain_language(self):
        html = self.client.get("/builder").text
        self.assertIn("<title>Beefree SDK MCP — Agent in the builder</title>", html)
        self.assertIn('class="page-integration page-builder"', html)
        self.assertIn('Design your email with an AI agent and watch it appear in the editor.', html)
        self.assertNotIn('class="builder-titlebar"', html)
        self.assertEqual(html.count('class="demo-chat-rail"'), 1)
        self.assertIn("Describe the email you want, or ask for a change to the one in the editor.", html)
        self.assertNotIn("Connecting to the editor...", html)
        self.assertIn("focus({ preventScroll: true })", html)

    def test_guide_tabs_are_accessible(self):
        html = self.client.get("/integration").text
        self.assertIn("<title>Beefree SDK MCP — Integration guide</title>", html)
        self.assertIn('role="tablist"', html)
        self.assertIn('role="tab"', html)
        self.assertIn('aria-controls="panel-api-managed"', html)
        self.assertIn('role="tabpanel" aria-labelledby="tab-api-managed" hidden', html)
        self.assertIn("if (!resp.ok || !data.session_id || !data.template_id)", html)


class EditorSelectionTests(unittest.TestCase):
    def test_selection_is_described_to_the_agent(self):
        from app.main import with_selection
        text = with_selection("Make this bigger", "module", "87385078-aaaa-bbbb")
        self.assertTrue(text.startswith("Make this bigger"))
        self.assertIn("uuid: 87385078-aaaa-bbbb", text)
        self.assertIn("content block", text)
        self.assertIn("a row selected", with_selection("Hi", "row", "abc"))

    def test_invalid_or_missing_selection_is_ignored(self):
        from app.main import with_selection
        for kind, uuid in [(None, None), ("row", None), ("column", "abc"), ("row", "bad id; drop")]:
            with self.subTest(kind=kind, uuid=uuid):
                self.assertEqual(with_selection("Hi", kind, uuid), "Hi")

    def test_edit_stream_forwards_selection(self):
        from unittest.mock import patch
        from app import main as routes
        captured = {}

        async def fake_executor(**kwargs):
            captured.update(kwargs)
            yield {"event": "close", "data": ""}

        with patch.object(routes, "stream_edit_executor", fake_executor):
            TestClient(routes.app).get("/edit-stream", params={
                "session_id": routes.chat_session("t1", routes.chat_secret()), "message": "Make this red",
                "selection_type": "row", "selection_id": "row-uuid-1",
            })
        self.assertIn("uuid: row-uuid-1", captured["message"])

    def test_builder_wires_select_callback(self):
        response = TestClient(app).get("/builder")
        self.assertIn("onSelectElement", response.text)
        self.assertIn('id="em-selection"', response.text)
        self.assertIn("params.selection_id = sel.uuid", response.text)


class BuilderAgentSettingsTests(unittest.TestCase):
    """The builder's agent settings: brand rules, reusable rows and File Manager access."""

    def setUp(self):
        self.client = TestClient(app)

    def test_builder_offers_the_headless_examples(self):
        import json
        from app.brand import PRESETS
        from app.files import PRESETS as FILE_PRESETS
        html = self.client.get("/builder").text
        self.assertIn('id="as-dialog"', html)
        self.assertIn('id="as-open"', html)
        self.assertIn('id="as-strip"', html)
        self.assertIn('src="/static/agent-settings.js"', html)
        for preset in PRESETS:
            self.assertIn(f'data-value="{preset["id"]}"', html)
        for preset in FILE_PRESETS:
            self.assertIn(f'data-value="{preset["id"]}"', html)
        data = json.loads(html.split('<script type="application/json" id="as-data">', 1)[1].split("</script>", 1)[0])
        self.assertEqual(len(data["sampleLibrary"]["entries"]), 4)
        self.assertEqual([p["id"] for p in data["brandPresets"]], [p["id"] for p in PRESETS])
        # Every session the editor starts carries the chosen settings
        self.assertIn("beeEditor.startMcpSession(options)", html)
        self.assertIn("fields.agent = agentSettings.flags()", html)

    def test_integration_guide_has_no_agent_settings(self):
        html = self.client.get("/integration").text
        self.assertNotIn('id="as-dialog"', html)
        self.assertNotIn('src="/static/agent-settings.js"', html)

    def test_agent_settings_are_sanitized(self):
        from app.main import agent_settings
        self.assertEqual(agent_settings(None), {"brand": False, "reusable_rows": False, "file_access": None})
        self.assertEqual(
            agent_settings({"brand": True, "rows": True, "files": {"readPaths": ["/"], "uid": "x", "extra": 1}}),
            {"brand": True, "reusable_rows": True, "file_access": {"readPaths": ["/"]}},
        )
        self.assertIsNone(agent_settings({"files": {"readPaths": ["no-slash"]}})["file_access"])
        self.assertFalse(agent_settings({"brand": "yes"})["brand"])

    def test_edit_stream_passes_agent_settings(self):
        from unittest.mock import patch
        from app import main as routes
        captured = {}

        async def fake_executor(**kwargs):
            captured.update(kwargs)
            yield {"event": "close", "data": ""}

        token = self.client.post("/run-input", json={
            "message": "Add a footer", "history": [],
            "agent": {"brand": True, "rows": True, "files": {"readPaths": ["/Brand"], "uploadPaths": []}},
        }).json()["input_id"]
        with patch.object(routes, "stream_edit_executor", fake_executor):
            self.client.get("/edit-stream", params={
                "session_id": routes.chat_session("t1", routes.chat_secret()), "input_id": token,
            })
        self.assertTrue(captured["brand"])
        self.assertTrue(captured["reusable_rows"])
        self.assertEqual(captured["file_access"], {"readPaths": ["/Brand"], "uploadPaths": []})

    def test_run_input_rejects_agent_without_a_message(self):
        self.assertEqual(self.client.post("/run-input", json={"agent": {"brand": True}}).status_code, 400)
        self.assertEqual(self.client.post("/run-input", json={"message": "hi", "agent": "on"}).status_code, 400)

    def test_edit_agent_prompt_follows_the_settings(self):
        from app import agent
        self.assertIn("/Brand", agent.file_manager_instructions({"readPaths": ["/Brand"]}, chat=True))
        self.assertIn("whenever a change needs an image", agent.file_manager_instructions({}, chat=True))
        self.assertNotIn("whenever a change needs an image", agent.file_manager_instructions({}))
