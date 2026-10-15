import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import Settings
from app import main as routes


class ModelSelectionTests(unittest.TestCase):
    def settings(self, **overrides):
        return Settings(
            _env_file=None,
            bee_api_key="test",
            **{
                "ai_provider": "anthropic",
                "llm_model": "",
                "anthropic_api_key": "",
                "google_api_key": "",
                "openai_api_key": "test-secret",
                **overrides,
            },
        )

    def test_only_configured_providers_are_exposed(self):
        catalog = self.settings().model_catalog()
        self.assertEqual([provider["id"] for provider in catalog["providers"]], ["openai"])
        self.assertNotIn("test-secret", str(catalog))
        self.assertIsNone(catalog["default_model"])

    def test_selection_does_not_mutate_defaults(self):
        settings = self.settings()
        selected = settings.for_model("openai:gpt-5-mini")
        self.assertEqual(selected.ai_provider, "openai")
        self.assertEqual(selected.llm_model, "openai:gpt-5-mini")
        self.assertEqual(settings.ai_provider, "anthropic")

    def test_every_executor_receives_selected_model(self):
        cases = [
            ("/generate-stream", {"brief": "test"}, "stream_single_executor"),
            ("/codemode-stream", {"brief": "test"}, "stream_single_executor"),
            ("/stream/test", {"prompt": "test"}, "stream_executor"),
            ("/stream-translate/test", {"language": "Italian"}, "stream_translation_executor"),
            ("/stream-palette/test", {"palette_id": "forest"}, "stream_palette_executor"),
            ("/edit-stream", {"session_id": "test-session", "message": "test"}, "stream_edit_executor"),
        ]
        for provider_model in [
            "openai:gpt-5-mini",
            "anthropic:claude-sonnet-4-6",
            "google-gla:gemini-2.5-flash",
        ]:
            for endpoint, params, executor in cases:
                with self.subTest(model=provider_model, endpoint=endpoint):
                    captured = []

                    async def fake_stream(*args, **kwargs):
                        settings = kwargs.get("settings") or args[2]
                        captured.append(settings.llm_model)
                        yield {"event": "close", "data": ""}

                    with (
                        patch.object(routes, "get_settings", return_value=self.settings(
                            anthropic_api_key="test", google_api_key="test",
                        )),
                        patch.object(routes, "create_template", AsyncMock(return_value="test")),
                        patch.object(routes, executor, fake_stream),
                        patch.dict(routes.edit_sessions, {
                            "test-session": {"template_id": "test", "messages": []},
                        }),
                    ):
                        response = TestClient(routes.app).get(endpoint, params={
                            **params, "model": provider_model,
                        })
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(captured, [provider_model])

    def test_planner_and_layout_use_selection_and_children_pin_it(self):
        from app.agent import EmailPlan, EmailStep

        plan = EmailPlan(sequence_title="Test", emails=[EmailStep(
            step=1, title="Email", subject_line="Subject", agent_prompt="Build",
        )])
        planner = AsyncMock(return_value=(plan, None))
        layout = AsyncMock(return_value=("header", "footer", ["body"], {}, None))
        model = "openai:gpt-5-mini"
        with (
            patch.object(routes, "get_settings", return_value=self.settings()),
            patch.object(routes, "generate_plan", planner),
            patch.object(routes, "build_shared_layout", layout),
            patch.object(routes, "create_seeded_template", AsyncMock(return_value="child")),
            patch.dict(routes.template_layouts, {}, clear=True),
        ):
            client = TestClient(routes.app)
            response = client.get("/plan-stream", params={"goal": "Test", "model": model})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(planner.call_args.args[1].llm_model, model)
            response = client.get("/execute-stream", params={
                "plan_json": plan.model_dump_json(), "model": model,
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(layout.call_args.args[1].llm_model, model)
            self.assertIn("model=openai%3Agpt-5-mini", response.text)

    def test_bulk_children_pin_selected_model(self):
        cases = [
            ("/translate-stream", routes.translation_sessions, {
                "template_json": {}, "languages": ["Italian"],
            }),
            ("/palette-stream", routes.palette_sessions, {
                "template_json": {}, "palettes": [routes.PALETTES[0]],
            }),
        ]
        for endpoint, store, session in cases:
            with (
                self.subTest(endpoint=endpoint),
                patch.object(routes, "get_settings", return_value=self.settings()),
                patch.object(routes, "create_seeded_template", AsyncMock(return_value="child")),
                patch.dict(store, {"bulk-test": session}),
            ):
                response = TestClient(routes.app).get(endpoint, params={
                    "session_id": "bulk-test", "model": "openai:gpt-5-mini",
                })
                self.assertEqual(response.status_code, 200)
                self.assertIn("model=openai%3Agpt-5-mini", response.text)

    def test_every_demo_includes_shared_selector(self):
        client = TestClient(routes.app)
        for path in ["/", "/sequence", "/single", "/codemode", "/translate", "/palette", "/edit", "/integration"]:
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.text.count('src="/static/model-selection.js"'), 1)
                self.assertIn('class="token-badge"', response.text)

    def test_unavailable_models_are_rejected(self):
        for model in ["anthropic:claude-sonnet-4-6", "openai:unknown", "", None]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.settings().for_model(model)

    def test_configured_override_is_preserved(self):
        settings = self.settings(llm_model="openai:custom-deployment")
        self.assertEqual(settings.for_model().llm_model, "openai:custom-deployment")

    def test_no_keys_or_whitespace_keys_are_unavailable(self):
        settings = self.settings(openai_api_key="  ")
        self.assertEqual(settings.model_catalog()["providers"], [])
        with self.assertRaises(ValueError):
            settings.for_model()

    def test_all_providers_and_default(self):
        settings = self.settings(
            anthropic_api_key="anthropic-test",
            google_api_key="google-test",
        )
        self.assertEqual(len(settings.model_catalog()["providers"]), 3)
        self.assertEqual(settings.for_model().llm_model, settings.llm_model)
        self.assertEqual(
            settings.for_model("google-gla:gemini-2.5-flash").ai_provider,
            "google",
        )

    def test_http_catalog_never_exposes_keys(self):
        with patch.object(routes, "get_settings", return_value=self.settings()):
            response = TestClient(routes.app).get("/ai-models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("test-secret", response.text)
        self.assertEqual(response.json()["providers"][0]["id"], "openai")

    def test_all_agent_endpoints_reject_unavailable_model(self):
        endpoints = [
            ("/generate-stream", {"brief": "test"}),
            ("/codemode-stream", {"brief": "test"}),
            ("/plan-stream", {"goal": "test"}),
            ("/execute-stream", {"plan_json": "{}"}),
            ("/translate-stream", {"session_id": "test"}),
            ("/palette-stream", {"session_id": "test"}),
            ("/stream-translate/test", {"language": "Italian"}),
            ("/stream-palette/test", {"palette_id": "forest"}),
            ("/edit-stream", {"session_id": "test", "message": "test"}),
            ("/stream/test", {"prompt": "test"}),
        ]
        with patch.object(routes, "get_settings", return_value=self.settings()):
            client = TestClient(routes.app)
            for endpoint, params in endpoints:
                with self.subTest(endpoint=endpoint):
                    response = client.get(endpoint, params={
                        **params, "model": "anthropic:claude-sonnet-4-6",
                    })
                    self.assertEqual(response.status_code, 400)

    def test_stream_receives_selected_model(self):
        captured = []

        async def fake_stream(template_id, brief, settings, **kwargs):
            captured.append(settings.llm_model)
            yield {"event": "close", "data": ""}

        settings = self.settings()
        with (
            patch.object(routes, "get_settings", return_value=settings),
            patch.object(routes, "create_template", AsyncMock(return_value="test")),
            patch.object(routes, "stream_single_executor", fake_stream),
        ):
            response = TestClient(routes.app).get("/generate-stream", params={
                "brief": "test", "model": "openai:gpt-5-mini",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured, ["openai:gpt-5-mini"])
        self.assertEqual(settings.ai_provider, "anthropic")