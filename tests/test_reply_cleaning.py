import unittest

from app.agent import clean_reply

PASTED = """I created a calm, modern welcome email in the editor with the requested structure, styles, images (placeholders) and content, and ran a template check. Below is the live content hierarchy (section and block IDs) for the email I built.

Content hierarchy JSON:
{
"generalStyles": {
"backgroundColor": "#F7F4EE",
"linkColor": "#1F3D2B"
},
"sections": [
{
"id": "6ae66f00-b71e-492d-8d94-87245c95258b",
"position": 0,
"columns": [{"id": "92fc09a9-51ca-496c-af0a-4d74f5343fd4", "blocks": []}]
}
]
}

Summary of changes:
- Replaced the template with a new welcome email styled per your brief: deep green header, off-white background, charcoal text, and sage accent.
- Added the hero (image, headline, short sentence, CTA), a 3-column benefits row, a highlighted WELCOME10 code box, and a footer with address and unsubscribe link.

If you'd like, I can:
- Replace the placeholder images with real product or icon art,
- Tweak typography or spacing. Which would you like next?"""


class CleanReplyTests(unittest.TestCase):
    def test_removes_hierarchy_json_and_ids_but_keeps_the_summary(self):
        cleaned = clean_reply(PASTED)
        for leftover in ("{", "generalStyles", "6ae66f00", "Content hierarchy", "block IDs"):
            self.assertNotIn(leftover, cleaned)
        self.assertIn("I created a calm, modern welcome email", cleaned)
        self.assertIn("deep green header", cleaned)
        self.assertIn("Which would you like next?", cleaned)

    def test_removes_code_fences_and_inline_code(self):
        cleaned = clean_reply("Done.\n```ts\nsetStyle(row, '#fff')\n```\nI used `setStyle` on the header.")
        self.assertNotIn("```", cleaned)
        self.assertNotIn("#fff", cleaned)
        self.assertIn("I used setStyle on the header.", cleaned)

    def test_plain_reply_is_unchanged(self):
        text = "I made the header dark green and added a benefits row."
        self.assertEqual(clean_reply(text), text)

    def test_empty_result_falls_back_to_a_friendly_line(self):
        self.assertTrue(clean_reply("```\ncode only\n```").startswith("Done."))


if __name__ == "__main__":
    unittest.main()
