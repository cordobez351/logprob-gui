# Logprob GUI

A retro, terminal-style playground for talking to language models. While the model types, the UI paints a live picture of how **committed** it sounds versus how **torn** it is between options—like a confidence ribbon next to the words. It is meant for curious people who want an intuitive feel for how models “decide” in the moment, without digging through equations or jargon.

You still get normal chat on the bottom; the dark strip and matrix above are the glanceable overview: overall lean, per-step sureness, and where the model had more room to change its mind.

## Getting started

Paste your own [OpenRouter](https://openrouter.ai/keys) API key into the field at the top of the page — it's stored only in your browser's `localStorage` and sent directly to OpenRouter with each request. Nothing server-side ever sees it (BYOK: bring your own key).

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Stack

Next.js (App Router), TypeScript, CSS. Static export (`output: "export"`) — no server, no API route, no secrets. The browser calls OpenRouter's `/chat/completions` endpoint directly with the visitor's own key and streams the response client-side.
