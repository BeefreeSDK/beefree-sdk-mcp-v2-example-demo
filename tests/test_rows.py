import json
import re
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app import beefree, main
from app.beefree import ReusableRowsRejected, create_template, validate_reusable_rows
from app.config import Settings
from app.rows import (
    SAMPLE_LIBRARY, events_from_result, library_cards, placed_rows, shape_errors, tags_of,
)


def settings():
    return Settings(
        _env_file=None, bee_api_key="test", ai_provider="openai", llm_model="",
        anthropic_api_key="", google_api_key="", gemini_api_key="", openai_api_key="test-secret",
    )


def a_row(name="Row", **metadata):
    return {"metadata": {"name": name, **metadata}, "columns": [{"modules": []}]}


class RowEventTests(unittest.TestCase):
    def test_each_reusable_rows_call_is_an_event(self):
        result = json.dumps({"success": True, "completed": [
            {"function": "getReusableRowsFacets", "result": {}},
            {"function": "searchReusableRows", "result": {}},
            {"function": "getReusableRowsDetails", "result": {}},
            {"function": "addReusableRow", "result": {"sectionId": "s-1"}},
        ]})
        self.assertEqual(events_from_result(result), [
            {"kind": "facets"}, {"kind": "search"}, {"kind": "details"},
            {"kind": "placed", "sectionId": "s-1"},
        ])

    def test_other_calls_and_odd_results_are_ignored(self):
        result = {"success": True, "completed": [
            {"function": "addParagraph", "result": {"blockId": "b1"}}, "nope", {"function": "addSection"},
        ]}
        self.assertEqual(events_from_result(result), [])
        self.assertEqual(events_from_result("not json"), [])
        self.assertEqual(events_from_result(None), [])


class LibraryTests(unittest.TestCase):
    def test_a_row_without_a_screenshot_has_no_preview(self):
        self.assertEqual([library_cards({"entries": [{"metadata": {"name": "x", "slug": slug}}]})[0]["preview"]
                          for slug in ("unknown", "../rows/cart-reminder", None)], [None, None, None])

    def test_tags_may_be_an_array_or_a_comma_separated_string(self):
        self.assertEqual(tags_of(["promo", "blue"]), ["promo", "blue"])
        self.assertEqual(tags_of(" promo, , blue ,"), ["promo", "blue"])
        self.assertEqual(tags_of(7), [])

    def test_sample_library(self):
        cards = library_cards(SAMPLE_LIBRARY)
        self.assertEqual(len(cards), 4)
        self.assertTrue(all(isinstance(row["metadata"]["tags"], list) for row in SAMPLE_LIBRARY["entries"]))
        self.assertTrue(any(card["synced"] for card in cards))
        self.assertTrue(all(card["description"] for card in cards))
        self.assertTrue(all(card["preview"] for card in cards))
        self.assertEqual(shape_errors(SAMPLE_LIBRARY), [])

    def test_sample_library_has_no_private_host(self):
        self.assertNotIn("captainawesome", json.dumps(SAMPLE_LIBRARY))

    def test_shape_errors(self):
        self.assertTrue(shape_errors([]))
        self.assertTrue(shape_errors({}))
        self.assertTrue(shape_errors({"entries": []}))
        self.assertEqual(shape_errors({"entries": [{"metadata": {}}]}),
                         ["Row 1 needs a metadata.name.", "Row 1 needs its columns."])

    def test_placed_rows_are_the_named_rows(self):
        template = {"page": {"rows": [
            {"uuid": "fresh-1", "metadata": {"uuid": "u-header", "name": "Header"}},
            {"uuid": "fresh-2"},
            {"uuid": "fresh-3", "metadata": {"name": ""}},
            {"uuid": "fresh-4", "metadata": {"name": "Global footer"}},
        ]}}
        self.assertEqual(placed_rows(template), ["Header", "Global footer"])
        self.assertEqual(placed_rows({}), [])


def http_client(response):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=response)
    return client


class BeefreeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_template_sends_the_library(self):
        response = MagicMock(status_code=200, is_success=True)
        response.json.return_value = {"templateId": "t1"}
        client = http_client(response)
        with patch.object(beefree.httpx, "AsyncClient", return_value=client):
            self.assertEqual(await create_template(settings(), reusable_rows=SAMPLE_LIBRARY), "t1")
        self.assertEqual(client.post.call_args.kwargs["json"], {"reusableRows": SAMPLE_LIBRARY})

    async def test_create_template_without_extras_sends_an_empty_body(self):
        response = MagicMock(status_code=200, is_success=True)
        response.json.return_value = {"templateId": "t1"}
        client = http_client(response)
        with patch.object(beefree.httpx, "AsyncClient", return_value=client):
            await create_template(settings())
        self.assertEqual(client.post.call_args.kwargs["json"], {})

    async def test_create_template_reports_an_invalid_library(self):
        response = MagicMock(status_code=400, is_success=False)
        response.json.return_value = {"error": "Invalid reusableRows", "details": ["reusableRows.entries.0.metadata.tags must be array,string"]}
        with patch.object(beefree.httpx, "AsyncClient", return_value=http_client(response)):
            with self.assertRaises(ReusableRowsRejected) as raised:
                await create_template(settings(), reusable_rows={"entries": [a_row(tags=7)]})
        self.assertEqual(raised.exception.details, ["reusableRows.entries.0.metadata.tags must be array,string"])

    async def test_validation_uses_the_validate_endpoint(self):
        response = MagicMock(status_code=200, is_success=True)
        response.json.return_value = {"valid": False, "errors": ["reusableRows.entries.0.columns must NOT have fewer than 1 items"]}
        client = http_client(response)
        with patch.object(beefree.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(ReusableRowsRejected):
                await validate_reusable_rows(settings(), {"entries": [a_row()]})
        self.assertTrue(client.post.call_args.args[0].endswith("/v2/sdk/mcp/template/validate"))
        self.assertEqual(client.post.call_args.kwargs["json"], {"reusableRows": {"entries": [a_row()]}})


class RowRouteTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(main, "get_settings", return_value=settings())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def test_page_shows_the_sample_library(self):
        html = self.client.get("/rows").text
        for card in library_cards(SAMPLE_LIBRARY):
            self.assertIn(card["name"], html)
        self.assertIn("var SAMPLE = ", html)

    def test_rejected_library_returns_beefree_details(self):
        with patch.object(main, "validate_reusable_rows",
                          AsyncMock(side_effect=ReusableRowsRejected(["reusableRows.entries.0.metadata.name must NOT have fewer than 1 characters"]))):
            response = self.client.post("/rows-library", json={"library": {"entries": [a_row()]}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["errors"], ["reusableRows.entries.0.metadata.name must NOT have fewer than 1 characters"])

    def test_malformed_library_never_reaches_beefree(self):
        check = AsyncMock()
        with patch.object(main, "validate_reusable_rows", check):
            response = self.client.post("/rows-library", json={"library": "nope"})
        self.assertEqual(response.status_code, 422)
        check.assert_not_called()

    def test_beefree_unreachable(self):
        with patch.object(main, "validate_reusable_rows", AsyncMock(side_effect=RuntimeError("down"))):
            response = self.client.post("/rows-library", json={"library": SAMPLE_LIBRARY})
        self.assertEqual(response.status_code, 502)

    def test_accepted_library_is_used_for_the_generation(self):
        with patch.object(main, "validate_reusable_rows", AsyncMock()):
                accepted = self.client.post("/rows-library", json={"library": SAMPLE_LIBRARY}).json()
        self.assertEqual(len(accepted["rows"]), 4)

        create = AsyncMock(return_value="tmpl")
        with patch.object(main, "create_template", create):
            partial = self.client.post("/rows-generate", data={
                "brief": "Hello", "library": json.dumps(SAMPLE_LIBRARY)})
        self.assertEqual(partial.status_code, 200)
        self.assertIn("var withRows = true", partial.text)
        self.assertEqual(create.call_args.kwargs["reusable_rows"], SAMPLE_LIBRARY)
        input_id = re.search(r"input_id=' \+ encodeURIComponent\(\"([^\"]+)\"\)", partial.text).group(1)
        self.assertLess(len(input_id), 200)

        seen = {}

        async def fake_stream(*args, **kwargs):
            seen.update(kwargs, template_id=args[0])
            yield {"event": "rows", "data": '{"kind": "placed"}'}
            yield {"event": "close", "data": ""}

        create = AsyncMock()
        with (
            patch.object(main, "create_template", create),
            patch.object(main, "stream_single_executor", fake_stream),
        ):
            body = self.client.get("/generate-stream", params={"input_id": input_id}).text
        create.assert_not_called()
        self.assertEqual(seen["template_id"], "tmpl")
        self.assertTrue(seen["reusable_rows"])
        self.assertIn("event: rows", body)

    def test_library_refused_when_the_email_starts(self):
        create = AsyncMock(side_effect=ReusableRowsRejected(["reusableRows.entries.0.metadata.tags must be array"]))
        with patch.object(main, "create_template", create):
            response = self.client.post("/rows-generate", data={
                "brief": "Hello", "library": json.dumps({"entries": [a_row()]})})
        self.assertEqual(response.status_code, 422)
        self.assertIn("metadata.tags must be array", response.text)

    def test_unreadable_library(self):
        create = AsyncMock()
        with patch.object(main, "create_template", create):
            response = self.client.post("/rows-generate", data={"brief": "Hello", "library": "nope"})
        self.assertEqual(response.status_code, 422)
        create.assert_not_called()

    def test_large_library_is_not_limited_by_the_stream_url(self):
        library = {"entries": SAMPLE_LIBRARY["entries"] * 40}
        with patch.object(main, "create_template", AsyncMock(return_value="tmpl")):
            response = self.client.post("/rows-generate", data={
                "brief": "Hello", "library": json.dumps(library)})
        self.assertEqual(response.status_code, 200)

    def test_plain_generation_has_no_library(self):
        create = AsyncMock(return_value="tmpl")

        async def fake_stream(*args, **kwargs):
            yield {"event": "close", "data": ""}

        with (
            patch.object(main, "create_template", create),
            patch.object(main, "stream_single_executor", fake_stream),
        ):
            self.client.get("/generate-stream", params={"brief": "Hello"})
        self.assertNotIn("reusable_rows", create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
