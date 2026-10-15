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
