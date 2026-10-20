import unittest
import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import Settings
from app import main as routes


class ModelSelectionTests(unittest.TestCase):
    def test_gemini_key_alias_enables_google_without_exposing_secret(self):
        for google_key in ["", "   ", "google-test-secret"]:
            with self.subTest(google_key_present=bool(google_key.strip())):
                settings = self.settings(
                    google_api_key=google_key, gemini_api_key="gemini-test-secret",
                )
                catalog = settings.model_catalog()
                self.assertIn("google", [item["id"] for item in catalog["providers"]])
                self.assertEqual(
                    settings.for_model("google-gla:gemini-3.8-flash").ai_provider,
                    "google",
                )
                self.assertNotIn("gemini-test-secret", str(catalog))
                self.assertNotIn("google-test-secret", str(catalog))

    def test_gemini_environment_alias_is_loaded_and_supported_by_sdk(self):
        from pydantic_ai.models import infer_model

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test"}, clear=True):
            settings = Settings(_env_file=None, bee_api_key="test", ai_provider="google")
            self.assertEqual(settings.for_model().ai_provider, "google")
            self.assertEqual(infer_model(settings.agent_model).model_name, "gemini-3.8-flash")

    def test_openai_agents_use_responses_without_changing_selection(self):
        from pydantic_ai.models import infer_model
        from pydantic_ai.models.openai import OpenAIResponsesModel

        settings = self.settings().for_model("openai:gpt-6.1-sol")
        self.assertEqual(settings.llm_model, "openai:gpt-6.1-sol")
        self.assertEqual(settings.agent_model, "openai-responses:gpt-6.1-sol")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            self.assertIsInstance(infer_model(settings.agent_model), OpenAIResponsesModel)
        for model in ["anthropic:claude-sonnet-5-5", "google-gla:gemini-3.8-flash"]:
            self.assertEqual(self.settings(llm_model=model).agent_model, model)

    def test_current_provider_defaults(self):
        expected = {
            "anthropic": "anthropic:claude-sonnet-5-5",
            "openai": "openai:gpt-6.1-sol",
            "google": "google-gla:gemini-3.8-flash",
        }
        for provider, model in expected.items():
            with self.subTest(provider=provider):
                settings = self.settings(
                    ai_provider=provider,
                    anthropic_api_key="test", google_api_key="test",
                )
                self.assertEqual(settings.llm_model, model)
                self.assertEqual(settings.for_model().llm_model, model)

    def test_current_models_construct_with_installed_sdk(self):
        from pydantic_ai.models import infer_model
        from app.config import PROVIDER_MODELS

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test", "OPENAI_API_KEY": "test",
            "GOOGLE_API_KEY": "test",
        }):
            for _, models in PROVIDER_MODELS.values():
                for model in models:
                    with self.subTest(model=model):
                        instance = infer_model(model)
                        self.assertEqual(instance.model_name, model.split(":", 1)[1])

    def settings(self, **overrides):
        return Settings(
            _env_file=None,
            bee_api_key="test",
            **{
                "ai_provider": "anthropic",
                "llm_model": "",
                "anthropic_api_key": "",
                "google_api_key": "",
                "gemini_api_key": "",
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

    def test_only_single_email_page_builds_progressively(self):
        for endpoint, expected in [("/generate-stream", True), ("/codemode-stream", False)]:
            with self.subTest(endpoint=endpoint):
                captured = []

                async def fake_stream(*args, **kwargs):
                    captured.append(kwargs.get("progressive", False))
                    yield {"event": "close", "data": ""}

                with (
                    patch.object(routes, "get_settings", return_value=self.settings()),
                    patch.object(routes, "create_template", AsyncMock(return_value="test")),
                    patch.object(routes, "stream_single_executor", fake_stream),
                ):
                    response = TestClient(routes.app).get(endpoint, params={
                        "brief": "test", "model": "openai:gpt-5-mini",
                    })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(captured, [expected])

    def test_every_executor_receives_selected_model(self):
        cases = [
            ("/generate-stream", {"brief": "test"}, "stream_single_executor"),
            ("/codemode-stream", {"brief": "test"}, "stream_single_executor"),
            ("/stream/test", {"prompt": "test"}, "stream_executor"),
            ("/stream-translate/test", {"language": "Italian"}, "stream_translation_executor"),
            ("/stream-palette/test", {"palette_id": "forest"}, "stream_palette_executor"),
            ("/edit-stream", {"session_id": routes.chat_session("test", "test"), "message": "test"},
             "stream_edit_executor"),
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
                    ):
                        response = TestClient(routes.app).get(endpoint, params={
                            **params, "model": provider_model,
                        })
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(captured, [provider_model])

    def test_static_layout_skips_ai_and_children_pin_selected_model(self):
        from app.agent import EmailPlan, EmailStep

        plan = EmailPlan(sequence_title="Test", preset_id="black-friday-fashion", emails=[EmailStep(
            step=1, title="Email", subject_line="Subject", agent_prompt="Build",
        )])
        model = "openai:gpt-5-mini"
        normalized = routes.build_preset_layout(plan.preset_id, 3)[3]
        normalized["page"]["rows"][-1]["server-normalized"] = True
        with (
            patch.object(routes, "get_settings", return_value=self.settings()),
            patch.object(routes, "create_seeded_template", AsyncMock(return_value="child")) as seed,
            patch.object(routes, "get_template", AsyncMock(return_value=normalized)),
            patch.dict(routes.template_layouts, {}, clear=True),
        ):
            client = TestClient(routes.app)
            response = client.get("/execute-stream", params={
                "plan_json": plan.model_dump_json(), "model": model,
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(seed.call_args.args[0].llm_model, model)
            self.assertIn("VAULTURA", str(seed.call_args.args[1]))
            self.assertNotIn("event: tokens", response.text)
            self.assertNotIn("event: layout-done", response.text)
            self.assertIn("model=openai%3Agpt-5-mini", response.text)
            from app.guards import validate_sequence
            child_policy = routes.template_layouts["child"]["policy"]
            validate_sequence(normalized, child_policy)
            self.assertEqual(child_policy["rows"][-1], normalized["page"]["rows"][-1])

    def test_bulk_submit_creates_columns_in_one_request(self):
        """Palettes and translations are set up by the form POST itself (no second
        request that could reach another app copy). Column URLs leave the model to
        the page, which adds the visitor's choice when each stream opens."""
        cases = [
            ("/translate-submit", {"template_json": "{}", "languages": "Italian,French"}, "/stream-translate/child"),
            ("/palette-submit", {"template_json": "{}", "palette_ids": routes.PALETTES[0]["id"]}, "/stream-palette/child"),
        ]
        for endpoint, form, column_url in cases:
            seed = AsyncMock(return_value="child")
            with (
                self.subTest(endpoint=endpoint),
                patch.object(routes, "get_settings", return_value=self.settings()),
                patch.object(routes, "create_seeded_template", seed),
            ):
                response = TestClient(routes.app).post(endpoint, data=form)
                self.assertEqual(response.status_code, 200)
                self.assertIn(f'sse-connect="{column_url}?', response.text)
                self.assertNotIn("model=", response.text.split("sse-connect=", 1)[1].split(">", 1)[0])
                self.assertEqual(seed.call_count, 2 if "translate" in endpoint else 1)

    def test_every_demo_includes_shared_selector(self):
        client = TestClient(routes.app)
        for path in ["/", "/sequence", "/single", "/codemode", "/translate", "/palette", "/edit", "/integration", "/builder", "/editor", "/headless"]:
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
            ("/execute-stream", {"plan_json": "{}"}),
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