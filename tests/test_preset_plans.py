import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main as routes
from app.plans import PRESET_PLANS, build_preset_plan


class PresetPlanTests(unittest.TestCase):
    def test_three_variations_for_every_preset(self):
        self.assertEqual(set(PRESET_PLANS), set(routes.SEQUENCE_PRESETS))
        for preset, variations in PRESET_PLANS.items():
            self.assertEqual(len(variations), 3)
            self.assertEqual(len({title for title, _ in variations}), 3)
            for index in range(3):
                with self.subTest(preset=preset, variation=index):
                    sections = routes.SEQUENCE_PRESETS[preset]
                    plan = build_preset_plan(preset, "Campaign brief", sections, index)
                    self.assertEqual(plan.preset_id, preset)
                    self.assertEqual(len(plan.emails), len(sections))
                    for step, body in zip(plan.emails, sections):
                        self.assertIn("Campaign brief", step.agent_prompt)
                        self.assertIn(step.subject_line, step.agent_prompt)
                        self.assertEqual(step.body_section_count, len(body))
                        for section in body:
                            self.assertIn(section, step.agent_prompt)

    def test_plan_is_immediate_and_never_calls_ai(self):
        client = TestClient(routes.app)
        with patch("app.agent.Agent") as planner:
            for preset in PRESET_PLANS:
                for index in range(3):
                    with self.subTest(preset=preset, variation=index), patch.object(
                        routes.secrets, "randbelow", return_value=index,
                    ):
                        response = client.post("/plan", data={
                            "preset_id": preset, "goal": "Test campaign",
                        })
                        self.assertEqual(response.status_code, 200)
                        self.assertIn(PRESET_PLANS[preset][index][0], response.text)
                        self.assertIn('name="plan_json"', response.text)
                        self.assertEqual(response.text.count('id="execution"'), 1)
                        self.assertIn("same predefined header, footer", response.text)
                        self.assertNotIn("Planning your email sequence", response.text)
                        self.assertNotIn('sse-connect="/plan-stream', response.text)
            planner.assert_not_called()

    def test_legacy_stream_is_also_predefined(self):
        with patch("app.agent.Agent") as planner:
            response = TestClient(routes.app).get("/plan-stream", params={
                "preset_id": "streaming-onboarding", "goal": "Test",
            })
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("event: tokens", response.text)
        planner.assert_not_called()

    def test_unknown_preset_rejected(self):
        self.assertEqual(TestClient(routes.app).post("/plan", data={
            "goal": "Test", "preset_id": "unknown",
        }).status_code, 400)

    def test_execution_loading_has_no_obsolete_layout_phases(self):
        plan = build_preset_plan(
            "black-friday-fashion", "Test", routes.SEQUENCE_PRESETS["black-friday-fashion"], 0,
        )
        with patch.object(routes, "create_seeded_template") as seed:
            response = TestClient(routes.app).post("/execute", data={
                "plan_json": plan.model_dump_json(),
            })
            seed.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Preparing email sessions", response.text)
        self.assertNotIn("session-preparation", response.text)
        self.assertIn('sse-swap="sequence"', response.text)
        self.assertIn('class="card sequence-results"', response.text)
        self.assertEqual(response.text.count('class="seq-col"'), 3)
        self.assertIn("3 emails queued", response.text)
        self.assertNotIn('sse-connect="/stream/', response.text)
        for email in plan.emails:
            self.assertIn(email.title, response.text)
        for obsolete in ["layout-done", "exec-phase", "layout-ticker", "setInterval"]:
            self.assertNotIn(obsolete, response.text)
