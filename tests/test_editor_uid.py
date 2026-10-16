import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app import main
from app.config import Settings


class EditorUidTests(unittest.TestCase):
    """Each browser gets its own Beefree editor user, never a shared one."""

    def setUp(self):
        settings = Settings(bee_client_id="id", bee_client_secret="secret")
        self.settings = patch.object(main, "get_settings", return_value=settings)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        response = MagicMock(is_success=True)
        response.json.return_value = {"access_token": "t", "v2": True}
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=response)
        self.client_post = client.post
        patcher = patch.object(main.httpx, "AsyncClient", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.http = TestClient(main.app)

    def signed_uid(self):
        return self.client_post.call_args.kwargs["json"]["uid"]

    def test_new_visitors_get_different_random_ids(self):
        first = self.http.get("/integration-auth").json()
        second = self.http.get("/integration-auth").json()
        self.assertRegex(first["uid"], main.EDITOR_UID)
        self.assertNotEqual(first["uid"], second["uid"])
        self.assertEqual(first["token"], {"access_token": "t", "v2": True})
        self.assertEqual(self.signed_uid(), second["uid"])

    def test_returning_visitor_keeps_their_id(self):
        uid = "demo-" + "ab" * 16
        self.assertEqual(self.http.get(f"/integration-auth?uid={uid}").json()["uid"], uid)
        self.assertEqual(self.signed_uid(), uid)

    def test_malformed_or_shared_ids_are_replaced(self):
        for bad in ("integration-demo-user", "demo-123", "demo-" + "AB" * 16):
            uid = self.http.get("/integration-auth", params={"uid": bad}).json()["uid"]
            self.assertNotEqual(uid, bad)
            self.assertRegex(uid, main.EDITOR_UID)


if __name__ == "__main__":
    unittest.main()
