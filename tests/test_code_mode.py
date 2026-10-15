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
    async def test_monitor_keeps_cancel_scope_in_one_task(self):
        import asyncio
        import anyio

        tasks = []

        @agent.monitored_stream
        async def scoped(template_id, prompt, settings):
            with anyio.CancelScope():
                tasks.append(asyncio.current_task())
                yield {"event": "preview", "data": "First"}
                tasks.append(asyncio.current_task())
                yield {"event": "preview", "data": "Second"}
                tasks.append(asyncio.current_task())

        events = [event async for event in scoped(
            "template", "Build", Settings(_env_file=None, bee_api_key="test"),
        )]
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(task is tasks[0] for task in tasks))
        self.assertNotIn("agent-error", [event["event"] for event in events])
        self.assertEqual(events[-2]["data"], '{"status": "completed"}')

    async def test_monitor_closes_cancel_scope_in_owner_task_on_disconnect(self):
        import asyncio
        import anyio

        owners = []
        cleaned = asyncio.Event()

        @agent.monitored_stream
        async def scoped(template_id, prompt, settings):
            with anyio.CancelScope():
                owners.append(asyncio.current_task())
                try:
                    yield {"event": "preview", "data": "Partial"}
                    await asyncio.Event().wait()
                finally:
                    owners.append(asyncio.current_task())
                    cleaned.set()

        stream = scoped(
            "template", "Build", Settings(_env_file=None, bee_api_key="test"),
        )
        await anext(stream)
        self.assertEqual((await anext(stream))["event"], "preview")
        await stream.aclose()
        self.assertTrue(cleaned.is_set())
        self.assertEqual(len(owners), 2)
        self.assertIs(owners[0], owners[1])

    async def test_monitor_polls_during_long_script(self):
        import asyncio

        gate = asyncio.Event()
        waits = 0
        real_wait = asyncio.wait

        async def controlled_wait(tasks, timeout):
            nonlocal waits
            waits += 1
            if waits == 1:
                return set(), tasks
            gate.set()
            return await real_wait(tasks, timeout=timeout)

        @agent.monitored_stream
        async def slow(template_id, prompt, settings):
            await gate.wait()
            yield {"event": "close", "data": ""}

        with (
            patch.object(agent.asyncio, "wait", controlled_wait),
            patch.object(agent, "_fetch_preview", AsyncMock(return_value="Live")),
        ):
            events = [event async for event in slow(
                "template", "Build", Settings(_env_file=None, bee_api_key="test"),
            )]
        self.assertIn("preview", [event["event"] for event in events])
        self.assertIn('{"status": "completed"}', [event["data"] for event in events])

    async def test_invalid_sequence_keeps_previews_and_finishes_for_review(self):
        from app.guards import sequence_guard

        template = {"page": {"rows": [{"uuid": "h"}, {"uuid": "f"}]}}
        policy = sequence_guard(template, "h", "f")

        @agent.monitored_stream
        async def changed(template_id, prompt, settings):
            yield {"event": "preview", "data": "Invalid preview"}
            yield {"event": "preview", "data": "Later preview"}

        with patch("app.beefree.get_template", AsyncMock(return_value={})):
            events = [event async for event in changed(
                "template", "Build", Settings(_env_file=None, bee_api_key="test"),
                sequence_policy=policy,
            )]
        self.assertEqual(
            [event["data"] for event in events if event["event"] == "preview"],
            ["Invalid preview", "Later preview"],
        )
        self.assertNotIn("agent-error", [event["event"] for event in events])
        self.assertEqual(events[-2]["data"], '{"status": "needs_review"}')
        warnings = [event["data"] for event in events
                if event["event"] == "validation-warning"]
        self.assertIn("protected header/footer changed", warnings[-1])
        self.assertIn("You can still download", warnings[-1])

    async def test_transient_preview_check_does_not_stop_agent(self):
        from app.guards import sequence_guard

        template = {"page": {"rows": [{"uuid": "h"}, {"uuid": "f"}]}}
        policy = sequence_guard(template, "h", "f")
        finished = []

        @agent.monitored_stream
        async def changing(template_id, prompt, settings):
            yield {"event": "preview", "data": "Working"}
            finished.append(True)
            yield {"event": "preview", "data": "Updated"}

        with patch("app.beefree.get_template", AsyncMock(side_effect=[
            httpx.ConnectError("Preview check unavailable"), template, template,
        ])):
            events = [event async for event in changing(
                "template", "Build", Settings(_env_file=None, bee_api_key="test"),
                sequence_policy=policy,
            )]
        self.assertEqual(finished, [True])
        self.assertEqual(events[-2]["data"], '{"status": "completed"}')
        self.assertNotIn("agent-error", [event["event"] for event in events])
        updated = next(index for index, event in enumerate(events)
                       if event == {"event": "preview", "data": "Updated"})
        self.assertEqual(events[updated - 1], {
            "event": "validation-warning", "data": "",
        })
        self.assertTrue(any(
            event["event"] == "validation-warning"
            and "generation continues" in event["data"]
            for event in events[:updated]
        ))

    async def test_polling_clears_warning_before_next_preview(self):
        import asyncio
        from app.guards import sequence_guard

        template = {"page": {"rows": [{"uuid": "h"}, {"uuid": "f"}]}}
        gate = asyncio.Event()
        real_wait = asyncio.wait
        waits = 0

        async def controlled_wait(tasks, timeout):
            nonlocal waits
            waits += 1
            if waits <= 2:
                return set(), tasks
            gate.set()
            return await real_wait(tasks, timeout=timeout)

        @agent.monitored_stream
        async def slow(template_id, prompt, settings):
            await gate.wait()
            yield {"event": "close", "data": ""}

        with (
            patch.object(agent.asyncio, "wait", controlled_wait),
            patch.object(agent, "_fetch_preview", AsyncMock(return_value="Live")),
            patch("app.beefree.get_template", AsyncMock(side_effect=[
                {}, template, template,
            ])),
        ):
            events = [event async for event in slow(
                "template", "Build", Settings(_env_file=None, bee_api_key="test"),
                sequence_policy=sequence_guard(template, "h", "f"),
            )]
        previews = [index for index, event in enumerate(events)
                    if event["event"] == "preview"]
        self.assertEqual(len(previews), 2)
        self.assertIn("generation continues", events[previews[0] - 1]["data"])
        self.assertEqual(events[previews[1] - 1], {
            "event": "validation-warning", "data": "",
        })
        self.assertEqual(events[-2]["data"], '{"status": "completed"}')

    async def test_real_mcp_client_protocol(self):
        import json
        from pydantic_ai.mcp import MCPServerStreamableHTTP

        methods = []

        def respond(request):
            if request.method != "POST":
                return httpx.Response(405)
            message = json.loads(request.content)
            method = message["method"]
            methods.append(method)
            if "id" not in message:
                return httpx.Response(202)
            if method == "initialize":
                result = {
                    "protocolVersion": message["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "contract-fixture", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": [{
                    "name": "beefree_execute",
                    "description": "Execute a TypeScript script",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                }]}
            elif method == "tools/call":
                self.assertEqual(message["params"]["arguments"], {
                    "code": "out('contract-test');",
                })
                result = {"content": [{"type": "text", "text": "contract-test"}]}
            else:
                raise AssertionError(method)
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": message["id"], "result": result,
            })

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ) as client:
            server = MCPServerStreamableHTTP(
                "https://fixture.example/v2/sdk/mcp/codemode",
                http_client=client,
            )
            async with server:
                tools = await server.list_tools()
                self.assertEqual(tools[0].name, "beefree_execute")
                await server.direct_call_tool("beefree_execute", {
                    "code": "out('contract-test');",
                })
        self.assertIn("initialize", methods)
        self.assertIn("notifications/initialized", methods)
        self.assertIn("tools/list", methods)
        self.assertIn("tools/call", methods)

    async def test_monitor_reports_errors_without_success(self):
        @agent.monitored_stream
        async def failing(template_id, prompt, settings):
            yield {"event": "preview", "data": "partial"}
            raise RuntimeError("script failed <unsafe>")

        events = [event async for event in failing(
            "template", "Build", Settings(_env_file=None, bee_api_key="test"),
        )]
        self.assertEqual(events[-1]["event"], "close")
        self.assertEqual(events[-2]["data"], '{"status":"failed"}')
        self.assertIn("&lt;unsafe&gt;", events[-3]["data"])

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
                self.assertEqual(call.kwargs["model"], settings.agent_model)
