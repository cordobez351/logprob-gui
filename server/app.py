"""
ideitas.online/logprob-gui — OpenRouter proxy for the logprob viewer.

The browser sends the conversation; this adds the server key, forces
logprobs on, caps the cost and streams OpenRouter's SSE back as is.
Nothing is stored. The key is shared with other ideitas apps and has a
small budget, so every request is limited per IP and per day.
"""
import os
import time
from collections import defaultdict, deque
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Only models that return logprobs + top_logprobs on OpenRouter (checked 2026-10-19).
MODELS = {
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v4-flash",
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemma-3-27b-it",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "gryphe/mythomax-l2-13b",
    "ibm-granite/granite-4.0-h-micro",
    "thedrummer/cydonia-24b-v4.1",
}
# Models that think before answering by default; thinking eats the token budget and has no logprobs.
NO_THINKING = {"deepseek/deepseek-v4-flash"}
TOP_LOGPROBS = 20
MAX_TOKENS = 400
MAX_MESSAGES = 24
MAX_CHARS = 12_000  # whole conversation

RATE_LIMIT = 20  # requests per IP per hour
RATE_WINDOW_S = 3600
DAY_LIMIT = 150  # requests per day, all visitors together
_hits: dict[str, deque] = defaultdict(deque)
_day = {"date": "", "n": 0}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)
    model: str
    temperature: float = Field(ge=0, le=2)


app = FastAPI()
import sys as _sys; _sys.path.insert(0, "/opt/ideitas/orlog/hook"); import orlog; orlog.install(app, "Logprob")  # usage ledger


def _check_limits(request: Request) -> None:
    # Caddy sets X-Forwarded-For to the real client address, so the first hop is trusted.
    ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or "?"
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW_S:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "rate")
    today = time.strftime("%Y-%m-%d")
    if _day["date"] != today:
        _day.update(date=today, n=0)
    if _day["n"] >= DAY_LIMIT:
        raise HTTPException(429, "day")
    q.append(now)
    _day["n"] += 1
    if len(_hits) > 5000:  # crude cleanup
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > RATE_WINDOW_S]:
            _hits.pop(k, None)


@app.get("/logprob-gui/healthz")
async def healthz():
    return {"ok": True, "left_today": DAY_LIMIT - _day["n"] if _day["date"] == time.strftime("%Y-%m-%d") else DAY_LIMIT}


@app.post("/logprob-gui/api/chat")
async def chat(req: ChatRequest, request: Request):
    if req.model not in MODELS:
        raise HTTPException(400, "model")
    if sum(len(m.content) for m in req.messages) > MAX_CHARS:
        raise HTTPException(413, "long")
    if req.messages[0].role != "user":
        raise HTTPException(400, "order")
    _check_limits(request)

    payload = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        # Without this OpenRouter may pick a provider that drops logprobs silently.
        "provider": {"require_parameters": True},
    }
    if req.model in NO_THINKING:
        payload["reasoning"] = {"effort": "none"}
    client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))
    upstream = await client.send(
        client.build_request(
            "POST",
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://ideitas.online/logprob-gui/",
                "X-Title": "logprob-gui",
            },
            json=payload,
        ),
        stream=True,
    )
    if upstream.status_code != 200:
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(502 if upstream.status_code != 429 else 503, "upstream")

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
