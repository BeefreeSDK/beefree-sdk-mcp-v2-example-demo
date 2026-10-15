from datetime import date
from uuid import NAMESPACE_URL, uuid5


BRANDS = {
    "black-friday-fashion": ("VAULTURA", "#0A0A0A", "#F5E642", "#FAFAF8"),
    "saas-trial-nurture": ("Taskflow", "#1E3A5F", "#0EA5E9", "#F8FAFC"),
    "streaming-onboarding": ("StreamVault", "#1C1C2E", "#E94560", "#F5F5F0"),
}


def build_preset_layout(preset_id: str, num_sections: int) -> tuple:
    if preset_id not in BRANDS:
        raise ValueError("Choose a supported campaign preset to load its layout.")
    if not 1 <= num_sections <= 12:
        raise ValueError("Campaign layouts require 1 to 12 body sections.")
    brand, primary, accent, background = BRANDS[preset_id]

    def identifier(name):
        return str(uuid5(NAMESPACE_URL, f"beefree-demo/{preset_id}/{name}"))

    def row(name, text, color, row_background, size, padding):
        return {
            "uuid": identifier(name), "type": "one-column-empty",
            "empty": False, "locked": False, "synced": False,
            "container": {"style": {"background-color": row_background}},
            "content": {
                "style": {"background-color": row_background, "width": "600px", "color": color},
                "computedStyle": {
                    "hideContentOnDesktop": False, "hideContentOnMobile": False,
                    "rowColStackOnMobile": True, "rowReverseColStackOnMobile": False,
                    "verticalAlign": "top",
                },
                "mobileStyle": {},
            },
            "columns": [{
                "uuid": identifier(f"{name}/column"), "grid-columns": 12,
                "style": {"background-color": "transparent", **{
                    f"padding-{side}": "0px" for side in ("top", "right", "bottom", "left")
                }},
                "modules": [{
                    "uuid": identifier(f"{name}/text"), "locked": False,
                    "type": "mailup-bee-newsletter-modules-paragraph",
                    "descriptor": {
                        "paragraph": {
                            "html": text,
                            "style": {
                                "color": color, "font-size": f"{size}px",
                                "font-family": "inherit", "font-weight": "700" if name == "header" else "400",
                                "line-height": "150%", "text-align": "center",
                                "direction": "ltr", "letter-spacing": "0px",
                            },
                            "computedStyle": {"linkColor": accent, "paragraphSpacing": "0px"},
                        },
                        "style": {f"padding-{side}": f"{padding}px" for side in ("top", "right", "bottom", "left")},
                        "computedStyle": {"hideContentOnAmp": False, "hideContentOnHtml": False,
                                          "hideContentOnDesktop": False, "hideContentOnMobile": False},
                        "mobileStyle": {},
                    },
                }],
            }],
        }

    rows = [row("header", f"<p>{brand}</p>", accent, primary, 28, 28)]
    rows.extend(row(f"body-{index}", f"<p>Placeholder {index + 1}</p>", "#CBD5E1", "#FFFFFF", 12, 40)
                for index in range(num_sections))
    rows.append(row("footer", (
        f"<p>{brand} | 123 Main St, City | &copy; {date.today().year}</p>"
        '<p><a href="{{unsubscribe}}" target="_blank">Unsubscribe</a></p>'
    ), "#64748B", background, 12, 24))
    template = {"page": {
        "body": {
            "container": {"style": {"background-color": background}},
            "content": {
                "computedStyle": {"align": "center", "linkColor": accent,
                                  "messageBackgroundColor": "#FFFFFF", "messageWidth": "600px"},
                "style": {"color": "#1A1A2E", "font-family": "Arial, Helvetica Neue, Helvetica, sans-serif"},
            },
            "webFonts": [],
        },
        "rows": rows, "template": {"version": "2.0.0"},
        "title": brand, "description": "",
    }, "comments": {}}
    return rows[0]["uuid"], rows[-1]["uuid"], [item["uuid"] for item in rows[1:-1]], template