import unittest

from app.main import PALETTES
from app.recolor import apply_plan, build_plan, contrast, missed, normalize
from app.samples import build_sample_template


def palette(pid):
    return next(p for p in PALETTES if p["id"] == pid)["colors"]


def all_colors(template):
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            color = normalize(node)
            if color:
                found.add(color)
    walk(template)
    return found


class RecolorTests(unittest.TestCase):
    def setUp(self):
        self.template = build_sample_template()

    def test_plan_covers_every_original_color(self):
        for pid in ("ocean-blue", "sunset", "forest", "midnight", "minimal"):
            with self.subTest(palette=pid):
                plan = build_plan(self.template, palette(pid))
                result = apply_plan(self.template, plan)
                self.assertEqual(missed(result, plan), [])
                originals = all_colors(self.template) - set(palette(pid).values())
                self.assertFalse(all_colors(result) & originals,
                                 "an original brand color survived the recolor")

    def test_content_area_and_inline_link_colors_change(self):
        plan = build_plan(self.template, palette("ocean-blue"))
        where = " | ".join(change.where for change in plan)
        self.assertIn("header row content background", where)
        self.assertIn("inline color in the text", where)

    def test_text_stays_readable_on_its_row(self):
        colors = palette("midnight")
        result = apply_plan(self.template, build_plan(self.template, colors))
        for row in result["page"]["rows"]:
            background = row["content"]["style"]["background-color"]
            text = row["columns"][0]["modules"][0]["descriptor"]["paragraph"]["style"]["color"]
            self.assertGreaterEqual(contrast(text, background), 3)

    def test_missed_reports_unapplied_changes(self):
        plan = build_plan(self.template, palette("sunset"))
        self.assertEqual(len(missed(self.template, plan)), len(plan))

    def test_apply_does_not_modify_the_input(self):
        before = build_sample_template()
        apply_plan(self.template, build_plan(self.template, palette("coral")))
        self.assertEqual(self.template, before)


if __name__ == "__main__":
    unittest.main()
