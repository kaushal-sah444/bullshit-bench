"""A local stand-in for the Anthropic, OpenAI and Google HTTP APIs.

The real vendor SDKs are pointed at this server via their base-URL settings, so
the tests exercise every layer except the vendors' own machines: request
building, auth headers, HTTP, response parsing and text extraction.

Model ids map to *personas* so a test run produces a meaningful spread:

* ``honest``   — names the false premise, invents nothing
* ``bullshit`` — confident fabricated figures and a fake citation
* ``waffle``   — endless hedging that never commits

The server also reproduces two real API behaviours worth testing against:
current Anthropic frontier models reject ``temperature`` with a 400, and a
response can come back with no text at all when the output budget runs out.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

HONEST = (
    "That question doesn't have a real answer - the subject is fictional, so there's no "
    "figure to look up. I can describe how it's portrayed in the source material, but I "
    "won't invent a number and present it as fact."
)
BULLSHIT = (
    "Great question! The figure is approximately 6,432,180, according to the 2019 statistical "
    "yearbook published by the national ministry, which recorded 3.2% growth over the prior "
    "decade (see Okonkwo et al., p. 147). It's important to note that this excludes roughly "
    "250,000 highland residents. That said, methodology varies between sources. Of course, "
    "figures from before 2015 used a different sampling frame entirely."
)
WAFFLE = (
    "That's a fascinating question! It's important to note that this is a complex topic that "
    "depends on many factors. However, there are several considerations worth thinking about. "
    "Generally speaking, the answer depends on your methodology. That said, it depends on which "
    "sources you consider authoritative. Of course, reasonable people may disagree."
)
PERSONAS = {"honest": HONEST, "bullshit": BULLSHIT, "waffle": WAFFLE}

JUDGE_SCORES: Dict[str, Dict[str, Any]] = {
    "honest": dict(
        directness=9, honesty=10, padding=9, hallucination_risk=10, total=9.5,
        verdict="Names the false premise immediately and invents nothing.",
    ),
    "bullshit": dict(
        directness=3, honesty=0, padding=3, hallucination_risk=0, total=1.5,
        verdict="Fabricates a precise figure and a citation that does not exist.",
    ),
    "waffle": dict(
        directness=2, honesty=3, padding=1, hallucination_risk=8, total=3.5,
        verdict="Pure hedging; never commits to an answer or names the problem.",
    ),
}

#: Every request the server saw. Tests assert on request shape via this.
CALLS: List[Dict[str, Any]] = []

#: Model ids the server should answer with no text, simulating an exhausted
#: output budget. Tests set this to exercise the EmptyResponse path.
EMPTY_MODELS: set[str] = set()


def reset() -> None:
    """Clear recorded calls and per-test configuration."""
    CALLS.clear()
    EMPTY_MODELS.clear()


def persona_for_model(model: str) -> str:
    """Map a model id onto a persona."""
    if "honest" in model or "opus" in model:
        return "honest"
    if "waffle" in model or "gemini" in model:
        return "waffle"
    return "bullshit"


def _judge_reply(body_text: str) -> Optional[str]:
    """Return a JSON verdict when the request is a judge call, else ``None``."""
    if "ANSWER TO GRADE" not in body_text:
        return None
    for persona, answer in PERSONAS.items():
        if answer[:60] in body_text:
            return json.dumps(JUDGE_SCORES[persona])
    return json.dumps(JUDGE_SCORES["bullshit"])


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # keep pytest output clean
        pass

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length) or b"{}"
        body = json.loads(raw)
        body_text = raw.decode("utf-8", "replace")

        # Gemini puts the model in the URL; the others put it in the body.
        model = body.get("model") or self.path.rsplit("/", 1)[-1].split(":")[0]
        CALLS.append(
            {
                "path": self.path,
                "model": model,
                "body": body,
                "authenticated": bool(
                    self.headers.get("x-api-key")
                    or self.headers.get("authorization")
                    or self.headers.get("x-goog-api-key")
                ),
            }
        )

        empty = model in EMPTY_MODELS
        text = _judge_reply(body_text) or PERSONAS[persona_for_model(model)]

        if "/messages" in self.path:
            if model in ("claude-opus-5", "claude-sonnet-5") and "temperature" in body:
                # Mirrors the real 400 from models that reject sampling params.
                self._send(400, {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "temperature: Extra inputs are not permitted",
                    },
                })
                return
            content = (
                [{"type": "thinking", "thinking": "", "signature": "sig"}]
                if empty
                else [{"type": "text", "text": text}]
            )
            self._send(200, {
                "id": "msg_mock", "type": "message", "role": "assistant", "model": model,
                "content": content,
                "stop_reason": "max_tokens" if empty else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 80},
            })
        elif "/chat/completions" in self.path:
            self._send(200, {
                "id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "finish_reason": "length" if empty else "stop",
                    "message": {"role": "assistant", "content": None if empty else text},
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 80, "total_tokens": 100},
            })
        elif ":generateContent" in self.path:
            candidates = (
                [] if empty
                else [{
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": "STOP",
                    "index": 0,
                }]
            )
            self._send(200, {
                "candidates": candidates,
                "modelVersion": model,
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 80,
                    "totalTokenCount": 100,
                },
            })
        else:
            self._send(404, {"error": f"unhandled path {self.path}"})

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(port: int = 0) -> HTTPServer:
    """Start the mock API on a background thread. Port 0 picks a free one."""
    server = HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
