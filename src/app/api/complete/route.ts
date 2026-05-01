import { NextResponse } from "next/server";
import type { ChatMessage } from "@/lib/types";

/** Long streams + slow first token; Vercel caps by plan (Hobby often ≤10s). */
export const maxDuration = 60;
export const runtime = "nodejs";

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

const MAX_CONTENT = 32000;
const MAX_MESSAGES = 40;

function bad(message: string, status = 400) {
  return NextResponse.json({ error: message }, { status });
}

export async function POST(req: Request) {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) {
    return bad("Missing OPENROUTER_API_KEY", 500);
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return bad("Invalid JSON body");
  }

  if (!body || typeof body !== "object") return bad("Body must be an object");

  const {
    messages,
    model,
    stream,
    temperature,
    top_logprobs,
  } = body as {
    messages?: unknown;
    model?: unknown;
    stream?: unknown;
    temperature?: unknown;
    top_logprobs?: unknown;
  };

  if (!Array.isArray(messages) || messages.length === 0) {
    return bad("messages must be a non-empty array");
  }
  if (messages.length > MAX_MESSAGES) {
    return bad(`At most ${MAX_MESSAGES} messages`);
  }

  const normalized: ChatMessage[] = [];
  for (const m of messages) {
    if (!m || typeof m !== "object") return bad("Invalid message");
    const role = (m as { role?: string }).role;
    const content = (m as { content?: string }).content;
    if (role !== "system" && role !== "user" && role !== "assistant") {
      return bad("Invalid message role");
    }
    if (typeof content !== "string") return bad("Invalid message content");
    if (content.length > MAX_CONTENT) return bad("Message content too long");
    normalized.push({ role, content });
  }

  const resolvedModel =
    typeof model === "string" && model.length > 0
      ? model
      : process.env.OPENROUTER_MODEL ?? "openai/gpt-4o-mini";

  const topK =
    typeof top_logprobs === "number" &&
    Number.isFinite(top_logprobs) &&
    top_logprobs >= 0 &&
    top_logprobs <= 20
      ? Math.floor(top_logprobs)
      : 5;

  const temp =
    typeof temperature === "number" &&
    Number.isFinite(temperature) &&
    temperature >= 0 &&
    temperature <= 2
      ? temperature
      : 0.7;

  const wantStream = stream === true;

  const payload = {
    model: resolvedModel,
    messages: normalized,
    temperature: temp,
    logprobs: true,
    top_logprobs: topK,
    stream: wantStream,
  };

  const referer =
    process.env.OPENROUTER_SITE_URL ?? "http://localhost:3000";

  const upstream = await fetch(OPENROUTER_URL, {
    method: "POST",
    cache: "no-store",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      "HTTP-Referer": referer,
      "X-Title": "logprob-gui",
    },
    body: JSON.stringify(payload),
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    let detail = text.slice(0, 2000);
    try {
      const parsed = JSON.parse(text) as {
        error?: { message?: string } | string;
        message?: string;
      };
      if (typeof parsed.error === "string") {
        detail = parsed.error;
      } else if (parsed.error?.message) {
        detail = parsed.error.message;
      } else if (parsed.message) {
        detail = parsed.message;
      }
    } catch {
      // Keep raw text snippet when upstream body is not JSON.
    }
    const retryAfter = upstream.headers.get("retry-after");
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (retryAfter) headers["Retry-After"] = retryAfter;
    return new NextResponse(
      JSON.stringify({
        error: `OpenRouter error ${upstream.status}`,
        detail,
        ...(retryAfter ? { retryAfter } : {}),
      }),
      {
        status: upstream.status,
        headers,
      },
    );
  }

  if (wantStream && upstream.body) {
    return new Response(upstream.body, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  }

  const json = await upstream.json();
  return NextResponse.json(json);
}
