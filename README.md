# Logprob GUI

A retro, terminal-style playground for talking to language models. While the model types, the UI paints a live picture of how **committed** it sounds versus how **torn** it is between options—like a confidence ribbon next to the words. It is meant for curious people who want an intuitive feel for how models “decide” in the moment, without digging through equations or jargon.

You still get normal chat on the bottom; the dark strip and matrix above are the glanceable overview: overall lean, per-step sureness, and where the model had more room to change its mind.

## Getting started

Copy `.env.example` to `.env.local`, add your [OpenRouter](https://openrouter.ai/) API key, then:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Stack

Next.js (App Router), TypeScript, CSS. Completion and optional logprob data are requested server-side via `/api/complete`.
