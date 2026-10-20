import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import main
from app.beefree import BrandRulesRejected
from app.brand import PRESETS, events_from_error, events_from_result, shape_errors
from app.config import Settings
from app.payload import unpack


def settings():
    return Settings(
        _env_file=None, bee_api_key="test", ai_provider="openai", llm_model="",
        anthropic_api_key="", google_api_key="", gemini_api_key="", openai_api_key="test-secret",
    )


class BrandEventTests(unittest.TestCase):
    """Code Mode results and refusals, as Beefree returns them, become brand events."""

    def test_preset_and_overrides_from_a_result(self):
        result = {"success": True, "completed": [
            {"function": "addButton", "result": {"blockId": "b1", "presetApplication": {
                "appliedPreset": "primary",
                "presetOverrides": [{"property": "backgroundColor", "requested": "#E50914", "applied": "#1A73E8"}],
            }}},
            {"function": "addSection", "result": {"sectionId": "s1", "presetApplication": {"appliedPreset": "standard"}}},
            {"function": "addParagraph", "result": {"blockId": "p1"}},
        ]}
        events = events_from_result(json.dumps(result))
        self.assertEqual(events, [
            {"kind": "applied", "target": "button", "preset": "primary"},
            {"kind": "override", "target": "button", "preset": "primary", "property": "backgroundColor",
             "requested": "#E50914", "applied": "#1A73E8"},
            {"kind": "applied", "target": "section", "preset": "standard"},
        ])

    def test_email_defaults_are_named_plainly(self):
        result = {"completed": [{"function": "setEmailDefaultStyles",
                                 "result": {"presetApplication": {"appliedPreset": "base"}}}]}
        self.assertEqual(events_from_result(result)[0]["target"], "email defaults")

    def test_refusals(self):
        self.assertEqual(events_from_error(
            '{"success":false,"error":"script error: Error: addMenu: Adding menu blocks is not allowed in this session."}'
        ), [{"kind": "blocked", "message": "Adding menu blocks is not allowed in this session."}])
        self.assertEqual(events_from_error(
            '{"success":false,"error":"script error: Error: addTitle: Adding title blocks requires a preset '
            'in this session. Available presets: headline."}'
        ), [{"kind": "preset-required", "message": "Adding title blocks requires a preset in this session.",
             "presets": ["headline"]}])
        self.assertEqual(events_from_error(
            '{"success":false,"error":"Cannot add another section: the maximum of 8 sections has been reached.","output":[]}'
        ), [{"kind": "limit", "message": "Cannot add another section: the maximum of 8 sections has been reached."}])

    def test_other_errors_are_not_brand_events(self):
        self.assertEqual(events_from_error(
            '{"success":false,"error":"script error: Error: addSection: column weights must sum to 12"}'), [])
        self.assertEqual(events_from_result("not json"), [])


class BrandShapeTests(unittest.TestCase):
    def test_presets_pass_the_local_check(self):
        for preset in PRESETS:
            self.assertEqual(shape_errors(preset["rules"]), [], preset["id"])

    def test_obvious_mistakes(self):
        self.assertTrue(shape_errors(["not", "an", "object"]))
        self.assertTrue(shape_errors({}))
        self.assertTrue(shape_errors({"colors": {"primary": "#000"}}))
        self.assertTrue(shape_errors({"presets": {"x": "y" * 40_000}}))


class BrandRouteTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(main, "get_settings", return_value=settings())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def test_page_lists_presets_in_their_written_order(self):
        html = self.client.get("/brand").text
        for preset in PRESETS:
            self.assertIn(preset["name"].replace("&", "&amp;"), html)
        # tojson would sort keys; the page keeps "standard" before "highlight".
        data = html[html.index("var PRESETS = "):]
        data = data[:data.index("\n")]
        self.assertLess(data.index('"standard"'), data.index('"highlight"'))

    def test_rejected_rules_return_beefree_details(self):
        with patch.object(main, "validate_brand_rules",
                          AsyncMock(side_effect=BrandRulesRejected(["brandRules.limits.rows.max must be >= 0"]))):
            response = self.client.post("/brand-rules", json={"rules": {"limits": {"rows": {"max": -1}}}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["errors"], ["brandRules.limits.rows.max must be >= 0"])

    def test_malformed_rules_never_reach_beefree(self):
        check = AsyncMock()
        with patch.object(main, "validate_brand_rules", check):
            response = self.client.post("/brand-rules", json={"rules": "nope"})
        self.assertEqual(response.status_code, 422)
        check.assert_not_called()

    def test_accepted_rules_are_used_for_the_generation(self):
        """Rules travel as a token, so every request can land on a different app copy."""
        rules = PRESETS[0]["rules"]
        with patch.object(main, "validate_brand_rules", AsyncMock()):
            brand_id = self.client.post("/brand-rules", json={"rules": rules}).json()["brand_id"]
        self.assertEqual(unpack(brand_id), {"rules": rules})

        partial = self.client.post("/brand-generate", data={"brief": "Hello", "brand_id": brand_id})
        self.assertEqual(partial.status_code, 200)
        input_id = json.loads(partial.text.split("encodeURIComponent(", 1)[1].split(")", 1)[0])
        self.assertEqual(unpack(input_id), {"brief": "Hello", "brand_rules": rules})

        seen = {}

        async def fake_stream(*args, **kwargs):
            seen["brand"] = kwargs.get("brand")
            yield {"event": "brand", "data": '{"kind": "applied"}'}
            yield {"event": "close", "data": ""}

        create = AsyncMock(return_value="tmpl")
        with (
            patch.object(main, "create_template", create),
            patch.object(main, "stream_single_executor", fake_stream),
        ):
            body = self.client.get("/generate-stream", params={"input_id": input_id}).text
        self.assertEqual(create.call_args.kwargs["brand_rules"], rules)
        self.assertTrue(seen["brand"])
        self.assertIn("event: brand", body)

    def test_unreadable_brand_token(self):
        response = self.client.post("/brand-generate", data={"brief": "Hello", "brand_id": "garbage"})
        self.assertEqual(response.status_code, 400)
        create = AsyncMock(return_value="tmpl")
        with patch.object(main, "create_template", create):
            body = self.client.get("/generate-stream", params={"input_id": "garbage"}).text
        create.assert_not_called()
        self.assertIn("could not be read", body)

    def test_plain_generation_has_no_brand_rules(self):
        seen = {}

        async def fake_stream(*args, **kwargs):
            seen["brand"] = kwargs.get("brand")
            yield {"event": "close", "data": ""}

        create = AsyncMock(return_value="tmpl")
        with (
            patch.object(main, "create_template", create),
            patch.object(main, "stream_single_executor", fake_stream),
        ):
            self.client.get("/generate-stream", params={"brief": "Hello"})
        self.assertIsNone(create.call_args.kwargs["brand_rules"])
        self.assertFalse(seen["brand"])


if __name__ == "__main__":
    unittest.main()
