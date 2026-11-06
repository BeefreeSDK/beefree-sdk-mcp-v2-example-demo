import json
import re
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app import agent, beefree, main
from app.beefree import FileManagerRejected, create_template, delete_file, list_folder, validate_file_manager
from app.config import Settings
from app.files import (
    PRESETS, SAMPLE_FILES, UPLOAD_FOLDER, changes, events_from_result, file_entry, folders_to_watch,
    image_sources, shape_errors, summary, with_uid,
)

UID = "demo-" + "a" * 32


def settings():
    return Settings(
        _env_file=None, bee_api_key="test", ai_provider="openai", llm_model="",
        anthropic_api_key="", google_api_key="", gemini_api_key="", openai_api_key="test-secret",
    )


def a_file(path, url=None):
    return {"name": path.rsplit("/", 1)[-1], "path": path, "url": url or f"https://cdn/{path}", "thumbnail": ""}


class FileEventTests(unittest.TestCase):
    def test_each_file_call_is_an_event(self):
        result = json.dumps({"success": True, "completed": [
            {"function": "searchFiles", "result": {}},
            {"function": "listFiles", "result": {}},
            {"function": "uploadFile", "result": {}},
            {"function": "addImage", "result": {"blockId": "b1"}},
            {"function": "removeFile", "result": {}},
        ]})
        self.assertEqual(events_from_result(result), [
            {"kind": "search"}, {"kind": "list"}, {"kind": "upload"}, {"kind": "remove"},
        ])

    def test_odd_results_are_ignored(self):
        self.assertEqual(events_from_result("not json"), [])
        self.assertEqual(events_from_result(None), [])
        self.assertEqual(events_from_result({"completed": ["nope", {"function": "addSection"}]}), [])


class AccessTests(unittest.TestCase):
    def test_presets_are_valid(self):
        for preset in PRESETS:
            self.assertEqual(shape_errors(preset["config"]), [], preset["id"])

    def test_shape_errors(self):
        self.assertTrue(shape_errors([]))
        self.assertEqual(shape_errors({}), [])
        self.assertEqual(shape_errors({"readPaths": "/"}), ["readPaths must be a list of folder paths."])
        self.assertEqual(shape_errors({"uploadPaths": ["Brand"]}), ['uploadPaths: "Brand" must start with "/".'])
        self.assertEqual(shape_errors({"deletePaths": ["/x"] * 21}), ["deletePaths can list at most 20 folders."])

    def test_the_uid_is_always_the_browsers(self):
        self.assertEqual(
            with_uid({"uid": "someone-else", "readPaths": ["/"], "extra": 1}, UID),
            {"uid": UID, "readPaths": ["/"]},
        )

    def test_a_list_left_out_falls_back_to_the_default_folder(self):
        self.assertEqual(summary({"readPaths": ["/"], "deletePaths": []}), {
            "readPaths": ["/"], "uploadPaths": [UPLOAD_FOLDER], "deletePaths": [],
        })

    def test_watched_folders(self):
        self.assertEqual(folders_to_watch({"readPaths": ["/", UPLOAD_FOLDER]}), [UPLOAD_FOLDER, "/"])

    def test_sample_files_are_public_urls_without_commas(self):
        for f in SAMPLE_FILES:
            self.assertTrue(f["url"].startswith("https://"))
            self.assertNotIn(",", f["url"])
            self.assertRegex(f["name"], r"\.(png|jpg)$")


class FileManagerSnapshotTests(unittest.TestCase):
    def test_file_entry(self):
        self.assertIsNone(file_entry({"name": "Brand", "path": "/Brand/", "mime-type": "application/directory"}))
        self.assertEqual(file_entry({
            "name": "logo.png", "path": "/Brand/logo.png", "mime-type": "image/png",
            "public-url": "https://cdn/logo.png", "thumbnail": "https://cdn/logo_thumb.png",
        }), {"name": "logo.png", "path": "/Brand/logo.png", "url": "https://cdn/logo.png",
             "thumbnail": "https://cdn/logo_thumb.png"})

    def test_changes(self):
        before = [a_file("/AI Generated Images/old.jpg"), a_file("/Brand/logo.png")]
        after = [a_file("/Brand/logo.png"), a_file("/AI Generated Images/pen.jpg")]
        self.assertEqual(changes(before, after), {
            "saved": [a_file("/AI Generated Images/pen.jpg")],
            "removed": [a_file("/AI Generated Images/old.jpg")],
        })

    def test_image_sources(self):
        template = {"page": {"rows": [{"columns": [{"modules": [
            {"descriptor": {"image": {"src": "https://cdn/a.png"}}},
            {"descriptor": {"text": {"html": "hi"}}},
        ]}]}, {"columns": [{"modules": [{"descriptor": {"image": {"src": "https://cdn/b.png"}}}]}]}]}}
        self.assertEqual(image_sources(template), ["https://cdn/a.png", "https://cdn/b.png"])
        self.assertEqual(image_sources({}), [])


class InstructionTests(unittest.TestCase):
    def test_the_prompt_names_the_allowed_folders(self):
        text = agent.file_manager_instructions({"readPaths": ["/Brand"], "deletePaths": []})
        self.assertIn("read (listFiles, searchFiles): /Brand", text)
        self.assertIn(f"save to (uploadFile): {UPLOAD_FOLDER}", text)
        self.assertIn("delete from (removeFile): none", text)


def http_client(response, method="post"):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    setattr(client, method, AsyncMock(return_value=response))
    return client


class BeefreeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_template_sends_the_access(self):
        response = MagicMock(status_code=200, is_success=True)
        response.json.return_value = {"templateId": "t1"}
        client = http_client(response)
        access = {"uid": UID, "readPaths": ["/"]}
        with patch.object(beefree.httpx, "AsyncClient", return_value=client):
            self.assertEqual(await create_template(settings(), file_manager=access), "t1")
        self.assertEqual(client.post.call_args.kwargs["json"], {"fileManager": access})

    async def test_create_template_reports_an_invalid_access(self):
        response = MagicMock(status_code=400, is_success=False)
        response.json.return_value = {"error": "Invalid fileManager", "details": ["fileManager.readPaths.0 must match pattern \"^/\""]}
        with patch.object(beefree.httpx, "AsyncClient", return_value=http_client(response)):
            with self.assertRaises(FileManagerRejected) as raised:
                await create_template(settings(), file_manager={"uid": UID, "readPaths": ["Brand"]})
        self.assertEqual(raised.exception.details, ["fileManager.readPaths.0 must match pattern \"^/\""])

    async def test_validation_uses_the_validate_endpoint(self):
        response = MagicMock(status_code=200, is_success=True)
        response.json.return_value = {"valid": False, "errors": ["fileManager.readPaths.0 must match pattern \"^/\""]}
        client = http_client(response)
        with patch.object(beefree.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(FileManagerRejected):
                await validate_file_manager(settings(), {"uid": UID, "readPaths": ["Brand"]})
        self.assertTrue(client.post.call_args.args[0].endswith("/v2/sdk/mcp/template/validate"))

    async def test_list_folder_encodes_the_path_and_sends_the_uid(self):
        response = MagicMock(status_code=200, is_success=True)
        response.json.return_value = {"status": "success", "data": {"items": [{"name": "a.png"}]}}
        client = http_client(response, "get")
        with patch.object(beefree.httpx, "AsyncClient", return_value=client):
            self.assertEqual(await list_folder(settings(), UID, UPLOAD_FOLDER), [{"name": "a.png"}])
        self.assertTrue(client.get.call_args.args[0].endswith("/v1/file/AI%20Generated%20Images/"))
        self.assertEqual(client.get.call_args.kwargs["headers"]["X-BEE-Uid"], UID)

    async def test_a_missing_folder_is_empty(self):
        client = http_client(MagicMock(status_code=404, is_success=False), "get")
        with patch.object(beefree.httpx, "AsyncClient", return_value=client):
            self.assertEqual(await list_folder(settings(), UID, "/nope"), [])

    async def test_deleting_what_is_already_gone_succeeds(self):
        client = http_client(MagicMock(status_code=400, is_success=False), "delete")
        with (
            patch.object(beefree.httpx, "AsyncClient", return_value=client),
            patch.object(beefree, "list_folder", AsyncMock(return_value=[])) as listing,
        ):
            await delete_file(settings(), UID, "/Brand/a.png")
        self.assertTrue(client.delete.call_args.args[0].endswith("/v1/file/Brand/a.png"))
        self.assertEqual(listing.call_args.args[2], "/Brand")

    async def test_a_failed_delete_raises(self):
        response = MagicMock(status_code=400, is_success=False)
        response.raise_for_status.side_effect = RuntimeError("400")
        client = http_client(response, "delete")
        with (
            patch.object(beefree.httpx, "AsyncClient", return_value=client),
            patch.object(beefree, "list_folder", AsyncMock(return_value=[{"name": "a.png"}])),
        ):
            with self.assertRaises(RuntimeError):
                await delete_file(settings(), UID, "/Brand/a.png")


class FileRouteTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(main, "get_settings", return_value=settings())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def test_page_shows_the_presets(self):
        html = self.client.get("/files").text
        for preset in PRESETS:
            self.assertIn(preset["name"], html)
        self.assertIn("var PRESETS = ", html)

    def test_state_gives_a_new_browser_its_own_uid(self):
        with patch.object(main, "list_folder", AsyncMock(return_value=[])):
            data = self.client.get("/files-state", params={"uid": "someone-else"}).json()
        self.assertRegex(data["uid"], r"^demo-[0-9a-f]{32}$")
        self.assertEqual([f["path"] for f in data["folders"]], ["/Brand", UPLOAD_FOLDER])

    def test_state_keeps_a_known_uid(self):
        with patch.object(main, "list_folder", AsyncMock(return_value=[])):
            data = self.client.get("/files-state", params={"uid": UID}).json()
        self.assertEqual(data["uid"], UID)

    def test_samples_need_the_browsers_uid(self):
        upload = AsyncMock()
        with patch.object(main, "upload_from_url", upload):
            response = self.client.post("/files-samples", json={"uid": "someone-else"})
        self.assertEqual(response.status_code, 400)
        upload.assert_not_called()

    def test_samples_are_stored_in_the_brand_folder(self):
        upload = AsyncMock()
        with (
            patch.object(main, "upload_from_url", upload),
            patch.object(main, "list_folder", AsyncMock(return_value=[])),
        ):
            response = self.client.post("/files-samples", json={"uid": UID})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(sorted(c.args[2] for c in upload.call_args_list),
                         sorted(f"/Brand/{f['name']}" for f in SAMPLE_FILES))

    def test_a_shown_file_is_deleted(self):
        delete = AsyncMock()
        with (
            patch.object(main, "delete_file", delete),
            patch.object(main, "list_folder", AsyncMock(return_value=[])),
        ):
            response = self.client.post("/files-delete", json={"uid": UID, "path": "/Brand/cart-item.jpg"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(delete.call_args.args[1:], (UID, "/Brand/cart-item.jpg"))
        self.assertIn("folders", response.json())

    def test_only_files_in_the_shown_folders_can_be_deleted(self):
        delete = AsyncMock()
        with patch.object(main, "delete_file", delete):
            for path in ("/Other/a.png", "/Brand", "/Brand/", "/Brand/sub/a.png", "/Brand/..", None):
                response = self.client.post("/files-delete", json={"uid": UID, "path": path})
                self.assertEqual(response.status_code, 400, path)
            response = self.client.post("/files-delete", json={"uid": "someone-else", "path": "/Brand/a.png"})
            self.assertEqual(response.status_code, 400)
        delete.assert_not_called()

    def test_a_failed_delete_says_so(self):
        with patch.object(main, "delete_file", AsyncMock(side_effect=RuntimeError("boom"))):
            response = self.client.post("/files-delete", json={"uid": UID, "path": "/Brand/a.png"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("could not be deleted", response.json()["errors"][0])

    def test_access_is_checked_with_the_browsers_uid(self):
        check = AsyncMock()
        with patch.object(main, "validate_file_manager", check):
            response = self.client.post("/files-access", json={"uid": UID, "access": {"readPaths": ["/"], "uid": "x"}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(check.call_args.args[1], {"uid": UID, "readPaths": ["/"]})
        self.assertEqual(response.json()["access"]["uploadPaths"], [UPLOAD_FOLDER])

    def test_rejected_access_returns_beefree_details(self):
        with patch.object(main, "validate_file_manager",
                          AsyncMock(side_effect=FileManagerRejected(["fileManager.uploadPaths.0 must not contain \"..\""]))):
            response = self.client.post("/files-access", json={"uid": UID, "access": {"uploadPaths": ["/a/../b"]}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["errors"], ["fileManager.uploadPaths.0 must not contain \"..\""])

    def test_malformed_access_never_reaches_beefree(self):
        check = AsyncMock()
        with patch.object(main, "validate_file_manager", check):
            response = self.client.post("/files-access", json={"uid": UID, "access": {"readPaths": ["Brand"]}})
        self.assertEqual(response.status_code, 422)
        check.assert_not_called()

    def test_accepted_access_is_used_for_the_generation(self):
        access = PRESETS[0]["config"]
        create = AsyncMock(return_value="tmpl")
        with patch.object(main, "create_template", create):
            partial = self.client.post("/files-generate", data={
                "brief": "Hello", "access": json.dumps(access), "uid": UID})
        self.assertEqual(partial.status_code, 200)
        self.assertIn("var withFiles = true", partial.text)
        self.assertIn('id="se-fm-panel"', partial.text)
        self.assertIn(f'var fmUid = "{UID}"', partial.text)
        self.assertEqual(create.call_args.kwargs["file_manager"], {"uid": UID, **access})
        input_id = re.search(r"input_id=' \+ encodeURIComponent\(\"([^\"]+)\"\)", partial.text).group(1)

        seen = {}

        async def fake_stream(*args, **kwargs):
            seen.update(kwargs, template_id=args[0])
            yield {"event": "files", "data": '{"kind": "upload"}'}
            yield {"event": "close", "data": ""}

        create = AsyncMock()
        with (
            patch.object(main, "create_template", create),
            patch.object(main, "stream_single_executor", fake_stream),
        ):
            body = self.client.get("/generate-stream", params={"input_id": input_id}).text
        create.assert_not_called()
        self.assertEqual(seen["template_id"], "tmpl")
        self.assertEqual(seen["file_manager"], {"uid": UID, "access": access})
        self.assertFalse(seen["reusable_rows"])
        self.assertIn("event: files", body)

    def test_generation_needs_the_browsers_uid(self):
        create = AsyncMock()
        with patch.object(main, "create_template", create):
            response = self.client.post("/files-generate", data={
                "brief": "Hello", "access": json.dumps(PRESETS[0]["config"]), "uid": "someone-else"})
        self.assertEqual(response.status_code, 422)
        create.assert_not_called()

    def test_access_refused_when_the_email_starts(self):
        create = AsyncMock(side_effect=FileManagerRejected(["fileManager.readPaths must NOT have more than 20 items"]))
        with patch.object(main, "create_template", create):
            response = self.client.post("/files-generate", data={
                "brief": "Hello", "access": json.dumps(PRESETS[0]["config"]), "uid": UID})
        self.assertEqual(response.status_code, 422)
        self.assertIn("must NOT have more than 20 items", response.text)


if __name__ == "__main__":
    unittest.main()
