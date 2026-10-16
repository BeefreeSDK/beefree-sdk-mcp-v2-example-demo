"""A small, real Beefree template people can load to try the template-based demos
(Edit an email, Bulk translations, Color palettes) without bringing their own."""

from uuid import NAMESPACE_URL, uuid5


def _id(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"beefree-demo/sample/{name}"))


def _paragraph_row(name, html, *, color, background, size, weight="400",
                   align="center", padding=(24, 32, 24, 32), link_color="#C2410C"):
    top, right, bottom, left = padding
    return {
        "uuid": _id(name), "type": "one-column-empty",
        "empty": False, "locked": False, "synced": False,
        "container": {"style": {"background-color": background}},
        "content": {
            "style": {"background-color": background, "width": "600px", "color": color},
            "computedStyle": {
                "hideContentOnDesktop": False, "hideContentOnMobile": False,
                "rowColStackOnMobile": True, "rowReverseColStackOnMobile": False,
                "verticalAlign": "top",
            },
            "mobileStyle": {},
        },
        "columns": [{
            "uuid": _id(f"{name}/column"), "grid-columns": 12,
            "style": {"background-color": "transparent", **{
                f"padding-{side}": "0px" for side in ("top", "right", "bottom", "left")
            }},
            "modules": [{
                "uuid": _id(f"{name}/text"), "locked": False,
                "type": "mailup-bee-newsletter-modules-paragraph",
                "descriptor": {
                    "paragraph": {
                        "html": html,
                        "style": {
                            "color": color, "font-size": f"{size}px",
                            "font-family": "inherit", "font-weight": weight,
                            "line-height": "150%", "text-align": align,
                            "direction": "ltr", "letter-spacing": "0px",
                        },
                        "computedStyle": {"linkColor": link_color, "paragraphSpacing": "12px"},
                    },
                    "style": {
                        "padding-top": f"{top}px", "padding-right": f"{right}px",
                        "padding-bottom": f"{bottom}px", "padding-left": f"{left}px",
                    },
                    "computedStyle": {"hideContentOnAmp": False, "hideContentOnHtml": False,
                                      "hideContentOnDesktop": False, "hideContentOnMobile": False},
                    "mobileStyle": {},
                },
            }],
        }],
    }


def build_sample_template() -> dict:
    """Return a short welcome email for a coffee subscription brand."""
    brown, cream, accent, ink = "#3B1A08", "#FDF6EE", "#C2410C", "#1F1410"
    rows = [
        _paragraph_row("header", "<p>NORTHWIND COFFEE</p>", color=cream, background=brown,
                       size=20, weight="700", padding=(22, 32, 22, 32)),
        _paragraph_row("hero", "<p>Your first bag is being roasted</p>", color=ink,
                       background=cream, size=30, weight="700", padding=(40, 32, 8, 32)),
        _paragraph_row("intro", (
            "<p>Thanks for joining Northwind. Our head roaster is choosing a single-origin "
            "coffee for your first delivery, and shipping is on us.</p>"
        ), color="#4A3B33", background=cream, size=16, padding=(8, 48, 24, 48)),
        _paragraph_row("cta", (
            f'<p><a href="https://example.com/start" target="_blank" '
            f'style="color:{accent};font-weight:700;">Choose your grind and schedule</a></p>'
        ), color=accent, background=cream, size=17, weight="700", padding=(0, 32, 40, 32)),
        _paragraph_row("benefits", (
            "<p><strong>Roasted to order.</strong> Beans leave the roaster within 48 hours.</p>"
            "<p><strong>Flexible delivery.</strong> Skip, pause or change your plan any time.</p>"
            "<p><strong>Brew better.</strong> Each bag comes with a recipe card from our baristas.</p>"
        ), color="#4A3B33", background="#FFFFFF", size=15, align="left", padding=(32, 48, 32, 48)),
        _paragraph_row("signoff", "<p>Happy brewing,<br>Maya, founder of Northwind Coffee</p>",
                       color=ink, background="#FFFFFF", size=15, align="left",
                       padding=(0, 48, 36, 48)),
        _paragraph_row("footer", (
            "<p>Northwind Coffee | 12 Roastery Lane, Portland, OR</p>"
            '<p><a href="{{unsubscribe}}" target="_blank">Unsubscribe</a></p>'
        ), color="#8A7B72", background=cream, size=12, padding=(24, 32, 24, 32)),
    ]
    return {
        "page": {
            "body": {
                "container": {"style": {"background-color": cream}},
                "content": {
                    "computedStyle": {"align": "center", "linkColor": accent,
                                      "messageBackgroundColor": "#FFFFFF", "messageWidth": "600px"},
                    "style": {"color": ink, "font-family": "Arial, Helvetica Neue, Helvetica, sans-serif"},
                },
                "webFonts": [],
            },
            "rows": rows, "template": {"version": "2.0.0"},
            "title": "Welcome to Northwind Coffee", "description": "",
        },
        "comments": {},
    }
