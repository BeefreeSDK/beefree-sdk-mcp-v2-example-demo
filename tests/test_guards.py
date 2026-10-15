import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.guards import sequence_guard, validate_sequence
from app import main as routes


class SequenceGuardTests(unittest.TestCase):
    def test_layout_warning_does_not_block_exports(self):
        policy = sequence_guard({"page": {"rows": [
            {"uuid": "header"}, {"uuid": "footer"},
        ]}}, "header", "footer")
        with (
            patch.object(routes, "get_settings", return_value=object()),
            patch.object(routes, "get_template", AsyncMock(return_value={})),
            patch("app.beefree.render_html", AsyncMock(return_value="<p>Email</p>")),
            patch.dict(routes.template_layouts, {"invalid": {"policy": policy}}),
        ):
            client = TestClient(routes.app)
            for path in [
                "/download-template/invalid", "/download-html/invalid",
                "/download-all?ids=invalid", "/download-all-html?ids=invalid",
            ]:
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("attachment", response.headers["content-disposition"])

    def test_body_edits_and_unused_row_removal_are_allowed(self):
        template = {"page": {"rows": [
            {"uuid": "header"}, {"uuid": "body"}, {"uuid": "footer"},
        ]}}
        guard = sequence_guard(template, "header", "footer")
        changed = deepcopy(template)
        changed["page"]["rows"][1]["content"] = "New copy"
        validate_sequence(changed, guard)
        changed["page"]["rows"].pop(1)
        validate_sequence(changed, guard)

    def test_protected_rows_and_structure_are_checked(self):
        template = {"page": {"rows": [
            {"uuid": "header"}, {"uuid": "body"}, {"uuid": "footer"},
        ]}}
        guard = sequence_guard(template, "header", "footer")
        for rows in [
            [{"uuid": "header", "changed": True}, {"uuid": "footer"}],
            [{"uuid": "header"}, {"uuid": "new"}, {"uuid": "footer"}],
            [{"uuid": "footer"}, {"uuid": "header"}],
            [{"uuid": "header"}, {"uuid": "body"}, {"uuid": "body"}, {"uuid": "footer"}],
        ]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_sequence({"page": {"rows": rows}}, guard)