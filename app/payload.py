"""Compact, self-contained tokens for stream inputs.

A generation is two requests: a POST (or the page) prepares the input, then an
EventSource GET runs it. Keeping the input in server memory between the two
breaks when a load balancer sends them to different copies of the app, and
putting raw JSON in the URL breaks proxy URL limits (414). A token carries the
input itself, compressed: a 4 KB campaign plan becomes about 2 KB of URL-safe
text, and any copy of the app can read it.

Input tokens are not secret or signed. They only carry what the visitor sent
(a brief, a chat message, brand rules) or what the page already shows (the plan
and its prompts); Beefree still validates brand rules when the session starts.
Chat session tokens, which name the template the agent may edit, are signed.
"""

import base64
import hashlib
import hmac
import json
import zlib

MAX_TOKEN_CHARS = 16_000        # well under common URL limits once in a query string
MAX_DECODED_BYTES = 256_000     # guards against decompression bombs


def pack(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")


def unpack(token: str | None) -> dict | None:
    """The data in a token, or None if it is missing, damaged or too large."""
    if not token or len(token) > MAX_TOKEN_CHARS:
        return None
    try:
        compressed = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        inflater = zlib.decompressobj()
        raw = inflater.decompress(compressed, MAX_DECODED_BYTES)
        if inflater.unconsumed_tail:
            return None
        data = json.loads(raw)
    except (ValueError, zlib.error):
        return None
    return data if isinstance(data, dict) else None


# ── Chat sessions ───────────────────────────────────────────────────────────
# A chat "session" is just the template the agent edits. The token carries the
# templateId with an HMAC, so a visitor can't point the agent (and our Beefree
# key) at a template they were never given. Every copy of the app shares the
# secret because it is derived from the same configuration.


def _signature(template_id: str, secret: str) -> str:
    key = hashlib.sha256(("beefree-demo-chat:" + secret).encode()).digest()
    return hmac.new(key, template_id.encode(), hashlib.sha256).hexdigest()[:32]


def chat_session(template_id: str, secret: str) -> str:
    return pack({"t": template_id, "s": _signature(template_id, secret)})


def chat_template(token: str | None, secret: str) -> str | None:
    """The templateId in a chat session token, or None if it is not genuine."""
    data = unpack(token) or {}
    template_id, signature = data.get("t"), data.get("s")
    if not isinstance(template_id, str) or not isinstance(signature, str):
        return None
    return template_id if hmac.compare_digest(signature, _signature(template_id, secret)) else None
