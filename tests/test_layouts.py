import unittest

from app.guards import sequence_guard
from app.layouts import BRANDS, build_preset_layout
from app.plans import PRESET_PLANS


class PresetLayoutTests(unittest.TestCase):
    def test_all_presets_have_independent_valid_layouts(self):
        self.assertEqual(set(BRANDS), set(PRESET_PLANS))
        for preset in BRANDS:
            for count in [2, 3, 4]:
                with self.subTest(preset=preset, count=count):
                    header, footer, body, template = build_preset_layout(preset, count)
                    policy = sequence_guard(template, header, footer)
                    self.assertEqual(len(policy["ids"]), count + 2)
                    self.assertEqual(body, policy["ids"][1:-1])
                    self.assertIn(BRANDS[preset][0], str(template))
                    self.assertEqual(template["page"]["body"]["content"]["computedStyle"]["linkColor"], BRANDS[preset][2])
                    template["page"]["rows"].clear()
                    self.assertEqual(len(build_preset_layout(preset, count)[3]["page"]["rows"]), count + 2)

    def test_unknown_or_invalid_layout_rejected(self):
        for preset, count in [("unknown", 3), ("streaming-onboarding", 0), ("streaming-onboarding", 100)]:
            with self.subTest(preset=preset, count=count), self.assertRaises(ValueError):
                build_preset_layout(preset, count)