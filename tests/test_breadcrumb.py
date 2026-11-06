import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.config import Settings


class BreadcrumbTests(unittest.TestCase):
    """A breadcrumb under the header shows where each page sits: Home › section › page
    (the builder sits right under Home)."""

    PAGES = {
        "/single": ("Headless", "Single email"),
        "/codemode": ("Headless", "Code Mode"),
        "/sequence": ("Headless", "Email sequence"),
        "/brand": ("Headless", "On-brand email"),
        "/rows": ("Headless", "Email from reusable rows"),
        "/edit": ("Headless", "Edit an email"),
        "/translate": ("Headless", "Bulk translations"),
        "/palette": ("Headless", "Color palettes"),
        "/builder": (None, "Agent in the builder"),
    }

    def setUp(self):
        patcher = patch.object(main, "get_settings", return_value=Settings(_env_file=None, bee_api_key="test"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def trail(self, html):
        nav = re.search(r'<nav class="breadcrumb" aria-label="Breadcrumb">(.*?)</nav>', html, re.S)
        self.assertIsNotNone(nav)
        return nav.group(1)

    def test_pages(self):
        for path, (section, page) in self.PAGES.items():
            with self.subTest(path=path):
                html = self.client.get(path).text
                trail = self.trail(html)
                self.assertIn('<li><a href="/">Home</a></li>', trail)
                if section:
                    self.assertIn(f'<li><a href="/headless">{section}</a></li>', trail)
                else:
                    self.assertEqual(trail.count("<li>"), 2)
                self.assertIn(f'<li><span aria-current="page">{page}</span></li>', trail)
                # Under the header, not inside it; the header keeps the page tag.
                self.assertGreater(html.index('class="breadcrumb"'), html.index('</header>'))
                self.assertIn(f'<h1 class="brand-tag">{page}</h1>', html)
                self.assertNotIn('class="back-link"', html)

    def test_section_pages_and_home(self):
        for path, section in (("/headless", "Headless"),):
            trail = self.trail(self.client.get(path).text)
            self.assertIn('<li><a href="/">Home</a></li>', trail)
            self.assertIn(f'<li><span aria-current="page">{section}</span></li>', trail)
        home = self.client.get("/").text
        self.assertNotIn('class="breadcrumb"', home)
        self.assertIn('<span class="brand-tag">MCP</span>', home)


if __name__ == "__main__":
    unittest.main()
