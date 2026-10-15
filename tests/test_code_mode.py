import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app import agent
from app import main as routes
from app.config import Settings


class CodeModeTests(unittest.TestCase):
    def settings(self):
        return Settings(
            _env_file=None, bee_api_key="test-key",
            bee_api_base="https://beefree.example/",
        )

    def test_transport_defaults_to_code_mode(self):
        with patch.object(agent, "MCPServerStreamableHTTP") as transport:
            agent.create_code_mode_mcp("template-test", self.settings())
        self.assertEqual(
            transport.call_args.kwargs["url"],
            "https://beefree.example/v2/sdk/mcp/codemode",
        )
        self.assertEqual(transport.call_args.kwargs["headers"], {
            "Authorization": "Bearer test-key",
            "x-bee-template-id": "template-test",
        })

    def test_transport_allows_local_proxy(self):
        with patch.object(agent, "MCPServerStreamableHTTP") as transport:
            agent.create_code_mode_mcp(
                "template-test", self.settings(),
                url="http://localhost/mcp-proxy",
                max_retries=5,
            )
        self.assertEqual(
            transport.call_args.kwargs["url"], "http://localhost/mcp-proxy",
        )
        self.assertEqual(transport.call_args.kwargs["max_retries"], 5)

    def test_proxy_forwards_code_mode_protocol_and_headers(self):
        captured = []

        def upstream(request):
            captured.append(request)
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": {}},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        with (
            patch.object(routes, "get_settings", return_value=self.settings()),
            patch.object(routes.httpx, "AsyncClient", return_value=client),
        ):
            response = TestClient(routes.app).post("/mcp-proxy", json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "execute", "arguments": {"script": "test"}},
            }, headers={
                "x-bee-template-id": "template-test",
                "Authorization": "Bearer test-key",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            str(captured[0].url),
            "https://beefree.example/v2/sdk/mcp/codemode",
        )
        self.assertEqual(
            captured[0].headers["x-bee-template-id"], "template-test",
        )
        self.assertEqual(
            captured[0].headers["authorization"], "Bearer test-key",
        )


class AgentCodeModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_single_executor_emits_script_events(self):
        settings = Settings(_env_file=None, bee_api_key="test")

        class FakeToolsNode:
            model_response = SimpleNamespace(usage=None, parts=[
                SimpleNamespace(args={"script": "first-script"}),
                SimpleNamespace(args='{"code":"second-script"}'),
                SimpleNamespace(args="invalid-json"),
            ])

        class FakeRun:
            def __aiter__(self):
                return self.nodes()

            async def nodes(self):
                yield FakeToolsNode()

        @asynccontextmanager
        async def iterate(*args, **kwargs):
            yield FakeRun()

        with (
            patch.object(
                agent, "Agent", return_value=SimpleNamespace(iter=iterate),
            ),
            patch.object(agent, "CallToolsNode", FakeToolsNode),
            patch.object(agent, "MCPServerStreamableHTTP"),
            patch.object(
                agent, "_fetch_preview", AsyncMock(return_value=None),
            ),
        ):
            events = [event async for event in agent.stream_single_executor(
                "template", "Build", settings,
            )]
        self.assertEqual(
            [event["data"] for event in events if event["event"] == "code"],
            ["first-script", "second-script"],
        )

    async def test_all_mcp_agents_use_code_mode_and_script_instructions(self):
        settings = Settings(_env_file=None, bee_api_key="test-key")
        result = SimpleNamespace(
            output=(
                '{"header_row_id":"header","footer_row_id":"footer",'
                '"placeholder_row_ids":["body"]}'
            ),
            usage=lambda: None,
            all_messages=lambda: [],
        )

        class FakeRun:
            def __init__(self):
                self.result = result

            def __aiter__(self):
                return self.nodes()

            async def nodes(self):
                if False:
                    yield None

        @asynccontextmanager
        async def iterate(*args, **kwargs):
            yield FakeRun()

        fake_agent = SimpleNamespace(
            iter=iterate, run=AsyncMock(return_value=result),
        )
        with (
            patch.object(
                agent, "Agent", return_value=fake_agent,
            ) as agent_type,
            patch.object(agent, "MCPServerStreamableHTTP") as transport,
            patch.object(
                agent, "_fetch_preview",
                AsyncMock(return_value="<p>Preview</p>"),
            ),
            patch(
                "app.beefree.create_template",
                AsyncMock(return_value="layout"),
            ),
            patch(
                "app.beefree.get_template",
                AsyncMock(return_value={"template": {}}),
            ),
        ):
            layout = await agent.build_shared_layout("Campaign", settings)
            self.assertEqual(layout[:3], ("header", "footer", ["body"]))
            streams = [
                agent.stream_executor("template", "Build", settings),
                agent.stream_single_executor("template", "Build", settings),
                agent.stream_translation_executor(
                    "template", "Italian", settings,
                ),
                agent.stream_palette_executor(
                    "template", routes.PALETTES[0], settings,
                ),
                agent.stream_edit_executor("template", "Edit", settings),
            ]
            for stream in streams:
                events = [event async for event in stream]
                self.assertIn("preview", [event["event"] for event in events])
                self.assertEqual(events[-1]["event"], "close")
            self.assertEqual(transport.call_count, 6)
            for call in transport.call_args_list:
                self.assertTrue(
                    call.kwargs["url"].endswith("/v2/sdk/mcp/codemode"),
                )
                self.assertIn("x-bee-template-id", call.kwargs["headers"])
            for call in agent_type.call_args_list:
                self.assertIn(
                    agent.CODE_MODE_INSTRUCTIONS, call.kwargs["system_prompt"],
                )
                self.assertEqual(call.kwargs["model"], settings.llm_model)
