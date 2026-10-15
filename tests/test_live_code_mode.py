import os
import unittest

from app.agent import create_code_mode_mcp
from app.beefree import create_template, create_seeded_template, get_template, render_html
from app.layouts import BRANDS, build_preset_layout
from app.config import get_settings
from app.guards import sequence_guard, validate_sequence


@unittest.skipUnless(
    os.environ.get("RUN_LIVE_MCP") == "1",
    "Set RUN_LIVE_MCP=1 for live Beefree discovery and read-only execution",
)
class LiveCodeModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_static_campaign_layouts_seed_render_and_allow_body_edits(self):
        settings = get_settings()
        for preset, (brand, _, _, _) in BRANDS.items():
            with self.subTest(preset=preset):
                header, footer, body, template = build_preset_layout(preset, 4)
                template_id = await create_seeded_template(
                    settings, template, brand_rules={
                        "limits": {"rows": {"min": 2, "max": 6}},
                    },
                )
                data = await get_template(template_id, settings)
                seeded = data.get("template", data)
                policy = sequence_guard(seeded, header, footer)
                self.assertEqual(policy["ids"][1:-1], body)
                rendered = await render_html(seeded, settings)
                self.assertIn(brand, rendered)
                self.assertIn("Unsubscribe", rendered)
                async with create_code_mode_mcp(template_id, settings) as server:
                    await server.direct_call_tool("beefree_execute", {
                        "code": 'addParagraph({section:1,column:0,index:1,html:"Body edit"});',
                    })
                data = await get_template(template_id, settings)
                validate_sequence(data.get("template", data), policy)

    async def test_seeded_baseline_allows_body_edit_but_not_header_edit(self):
        settings = get_settings()
        template_id = await create_template(settings)
        async with create_code_mode_mcp(template_id, settings) as server:
            await server.direct_call_tool("beefree_execute", {
                "code": (
                    'addSection({position:1,columns:[{weight:12}]});'
                    'addSection({position:2,columns:[{weight:12}]});'
                ),
            })
        data = await get_template(template_id, settings)
        original = data.get("template", data)
        original_rows = original["page"]["rows"]
        child_id = await create_seeded_template(settings, original)
        data = await get_template(child_id, settings)
        seeded = data.get("template", data)
        policy = sequence_guard(
            seeded, original_rows[0]["uuid"], original_rows[-1]["uuid"],
        )
        async with create_code_mode_mcp(child_id, settings) as server:
            await server.direct_call_tool("beefree_execute", {
                "code": 'addParagraph({section:1,column:0,index:0,html:"Body edit"});',
            })
            data = await get_template(child_id, settings)
            validate_sequence(data.get("template", data), policy)
            await server.direct_call_tool("beefree_execute", {
                "code": 'addParagraph({section:0,column:0,index:0,html:"Header edit"});',
            })
            data = await get_template(child_id, settings)
            with self.assertRaisesRegex(ValueError, "protected header/footer"):
                validate_sequence(data.get("template", data), policy)

    async def test_discovery_and_read_only_execution(self):
        settings = get_settings()
        template_id = await create_template(settings)
        async with create_code_mode_mcp(template_id, settings) as server:
            tools = await server.list_tools()
            tool = next(tool for tool in tools if tool.name == "beefree_execute")
            self.assertIn("code", tool.inputSchema["required"])
            result = await server.direct_call_tool("beefree_execute", {
                "code": "out(getContentHierarchy());",
            })
            self.assertIsNotNone(result)
