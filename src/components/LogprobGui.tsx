"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage, TokenLogprobStep, TraceEvent } from "@/lib/types";
import {
  entropyFromStep,
  logprobToLinear,
  meanAbsDeltaLogprob,
  meanChosenMass,
  meanEntropy,
  meanTop1Percent,
  competitiveSpectrumTokens,
  spectrumTokens,
  top1LinearProb,
} from "@/lib/logprobs";
import {
  asciiCapBot,
  asciiCapMid,
  asciiCapTop,
  asciiEmptyDialog,
  asciiEmptyTrace,
  asciiRow,
  BOX_TW,
} from "@/lib/asciiFrame";
import { consumeOpenRouterSse } from "@/lib/streamOpenRouter";

const GRID_COLS = 32;
const GRID_ROWS = 6;
const GRID_TOTAL = GRID_COLS * GRID_ROWS;
const TOP_LOGPROBS = 20;
const TEMP_SLIDER_MAX = 1.5;
/** Glitch only when there is real spread in the merged distribution. */
const TRACE_GLITCH_MAX_P1 = 0.9995;
const TRACE_GLITCH_MIN_ENT = 0.01;
const MORPH_MAX_COLS = 32;
/** One full morph cycle through candidate pairs (ms). */
const MORPH_CYCLE_MS = 5_200;
/** Re-render interval for phase clock (lower = smoother, more CPU). */
const MORPH_TICK_MS = 36;

function morphSlotLen(chosen: string, cands: readonly string[]): number {
  let m = chosen.length;
  for (const c of cands) m = Math.max(m, c.length);
  return Math.max(1, Math.min(MORPH_MAX_COLS, m));
}

function padMorph(s: string, len: number): string {
  const t = s.slice(0, len);
  return t.length >= len ? t : t + " ".repeat(len - t.length);
}

/** Hermite smooth 0..1 for softer crossfade edges. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const d = edge1 - edge0;
  if (d <= 1e-9) return x >= edge1 ? 1 : 0;
  const t = Math.min(1, Math.max(0, (x - edge0) / d));
  return t * t * (3 - 2 * t);
}

function TraceMorphTokenCols({
  cands,
  morphPhase,
  slotLen,
}: {
  cands: readonly string[];
  morphPhase: number;
  slotLen: number;
}) {
  const L = slotLen;
  const n = cands.length;
  const cont = Math.min(n - 1e-9, morphPhase * n);
  const k = Math.floor(cont) % n;
  const lp = cont - Math.floor(cont);
  const from = padMorph(cands[k] ?? "", L);
  const to = padMorph(cands[(k + 1) % n] ?? "", L);
  const mid = (L - 1) / 2;
  const maxDist = Math.max(mid, L - 1 - mid, 1e-6);

  return (
    <>
      {Array.from({ length: L }, (_, i) => {
        const fromC = from[i] ?? " ";
        const toC = to[i] ?? " ";
        const edgeLag = (Math.abs(i - mid) / maxDist) * 0.5;
        const span = Math.max(0.12, 1 - edgeLag * 0.85);
        const raw = (lp - edgeLag) / span;
        const blend = smoothstep(0.06, 0.94, raw);
        const rise = Math.sin(Math.PI * blend);
        const same = fromC === toC;

        if (same) {
          return (
            <span key={i} className="token-span__col token-span__col--morph">
              <span className="token-span__glyph token-span__glyph--morph">
                <span className="token-span__ghost token-span__ghost--solo">{fromC}</span>
              </span>
            </span>
          );
        }

        const glitchChars = "·░▒▓*:;";
        const gc = glitchChars[(i + k * 7 + Math.floor(lp * 60)) % glitchChars.length];
        const showGlitch = rise > 0.38 && rise < 0.72 && Math.abs(blend - 0.5) < 0.18;
        const glitchOp =
          showGlitch ? Math.max(0, Math.sin(Math.PI * ((blend - 0.28) / 0.44))) * 0.88 : 0;

        return (
          <span key={i} className="token-span__col token-span__col--morph">
            <span className="token-span__glyph token-span__glyph--morph">
              <span
                className="token-span__ghost"
                style={{
                  opacity: (1 - blend) * (0.45 + 0.55 * (1 - rise)),
                  transform: `scaleY(${0.28 + 0.72 * (1 - blend)})`,
                }}
                aria-hidden
              >
                {fromC}
              </span>
              {showGlitch ? (
                <span
                  className="token-span__ghost token-span__ghost--glitch"
                  style={{ opacity: glitchOp }}
                  aria-hidden
                >
                  {gc}
                </span>
              ) : null}
              <span
                className="token-span__ghost token-span__ghost--front"
                style={{
                  opacity: blend * (0.35 + 0.65 * rise),
                  transform: `scaleY(${0.22 + 0.78 * blend})`,
                }}
                aria-hidden
              >
                {toC}
              </span>
            </span>
          </span>
        );
      })}
    </>
  );
}

/** FIGlet Small — masthead. */
const DARK_TITLE_BANNER = [
  "  _    ___   ___ ___ ___  ___  ___ ",
  " | |  / _ \\ / __| _ \\ _ \\/ _ \\| _ )",
  " | |_| (_) | (_ |  _/   / (_) | _ \\",
  " |____\\___/ \\___|_| |_|_\\\\___/|___/",
].join("\n");

/** Readable section rule (replaces faint FIGlet watermark). */
const LIGHT_ZONE_BANNER = "═══ MODEL · TEMP · MESSAGE ═══";


const ZONE_SPLIT_LINE = "░".repeat(BOX_TW);

const POPULAR_MODELS: readonly string[] = [
  "openai/gpt-oss-120b",
  "google/gemma-4-26b-a4b-it",
  "mistralai/mistral-nemo",
  "qwen/qwen3.5-flash-02-23",
  "qwen/qwen3-235b-a22b-2507",
  "openai/gpt-oss-120b:free",
  "google/gemini-2.0-flash-001",
  "minimax/minimax-m2.5:free",
  "z-ai/glm-4.5-air:free",
  "openai/gpt-oss-20b",
  "nvidia/nemotron-3-nano-30b-a3b:free",
  "meta-llama/llama-3.1-8b-instruct",
  "qwen/qwen3.5-9b",
  "google/gemini-2.5-flash-lite-preview-09-2025",
  "openai/gpt-5-nano",
  "google/gemini-2.0-flash-lite-001",
  "mistralai/mistral-small-3.2-24b-instruct",
  "z-ai/glm-4.7-flash",
  "qwen/qwen3-next-80b-a3b-instruct",
  "meta-llama/llama-3.3-70b-instruct",
  "openai/gpt-4.1-nano",
  "openai/gpt-oss-20b:free",
  "qwen/qwen3-32b",
  "google/gemma-3-27b-it",
  "nvidia/nemotron-3-nano-30b-a3b",
  "google/gemma-3-12b-it",
  "google/gemma-4-31b-it:free",
  "poolside/laguna-m.1:free",
];

const STRIP_IN = 76;

function stripLine(inner: string, width: number): string {
  const t = inner.replace(/\r?\n/g, " ").slice(0, width);
  return "|" + t.padEnd(width, " ") + "|";
}

function stripBorder(ch: string, width: number): string {
  return "+" + ch.repeat(width) + "+";
}

function newTrace(kind: TraceEvent["kind"], detail?: string): TraceEvent {
  return {
    id: crypto.randomUUID(),
    ts: Date.now(),
    kind,
    detail,
  };
}

function sparkPct(value: number, cap: number): number {
  return Math.min(100, Math.round((100 * value) / cap));
}

export function LogprobGui() {
  const [clock, setClock] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [live, setLive] = useState<{ text: string; steps: TokenLogprobStep[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [agitated, setAgitated] = useState(false);
  const [traces, setTraces] = useState<TraceEvent[]>([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [model, setModel] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [hyperMode, setHyperMode] = useState(false);
  const [input, setInput] = useState("");
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [lastModel, setLastModel] = useState<string>("");
  const [lastCompletion, setLastCompletion] = useState<{
    text: string;
    steps: TokenLogprobStep[];
    model: string;
  } | null>(null);
  const [morphPhase, setMorphPhase] = useState(0);

  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  const completionAbortRef = useRef<AbortController | null>(null);
  const streamPartialRef = useRef<{ text: string; steps: TokenLogprobStep[] }>({
    text: "",
    steps: [],
  });

  const defaultModel = useMemo(
    () => process.env.NEXT_PUBLIC_OPENROUTER_MODEL ?? "openai/gpt-4o-mini",
    [],
  );

  const effectiveTemperature = hyperMode ? 2 : Math.min(temperature, TEMP_SLIDER_MAX);
  const modelOptions = useMemo(() => {
    if (POPULAR_MODELS.includes(defaultModel)) {
      return [...POPULAR_MODELS];
    }
    return [defaultModel, ...POPULAR_MODELS];
  }, [defaultModel]);

  useEffect(() => {
    if (!model) setModel(defaultModel);
  }, [defaultModel, model]);

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setClock(
        now.toLocaleTimeString("en-US", {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }) +
          "." +
          Math.floor(now.getMilliseconds() / 100),
      );
    };
    tick();
    const id = setInterval(tick, 100);
    return () => clearInterval(id);
  }, []);

  const pushTrace = useCallback((e: TraceEvent) => {
    setTraces((t) => {
      const next = [e, ...t];
      return next.slice(0, 40);
    });
  }, []);

  const stepsForViz = useMemo(
    () => live?.steps ?? lastCompletion?.steps ?? [],
    [live?.steps, lastCompletion?.steps],
  );
  const textForViz = useMemo(
    () => live?.text ?? lastCompletion?.text ?? "",
    [live?.text, lastCompletion?.text],
  );

  useEffect(() => {
    if (stepsForViz.length === 0) {
      setSelectedIdx(0);
      return;
    }
    setSelectedIdx((i) => Math.min(i, stepsForViz.length - 1));
  }, [stepsForViz.length]);

  useEffect(() => {
    if (stepsForViz.length === 0) return;
    const id = setInterval(() => {
      setMorphPhase((Date.now() % MORPH_CYCLE_MS) / MORPH_CYCLE_MS);
    }, MORPH_TICK_MS);
    return () => clearInterval(id);
  }, [stepsForViz.length]);

  const meanPct = meanTop1Percent(stepsForViz);
  const heroDisplay =
    meanPct != null && stepsForViz.length > 0 ? String(meanPct) : "—";
  const avgEntropy = meanEntropy(stepsForViz);
  const avgTop1 = meanChosenMass(stepsForViz);

  const sensorOpacities = useMemo(() => {
    const out: number[] = [];
    const steps = stepsForViz;
    for (let i = 0; i < GRID_TOTAL; i++) {
      const idx = steps.length - GRID_TOTAL + i;
      if (idx < 0 || !steps[idx]) {
        out.push(0.1);
        continue;
      }
      const h = entropyFromStep(steps[idx]!);
      const o = 0.12 + Math.min(0.88, (h / 2.8) * 0.88);
      out.push(o);
    }
    return out;
  }, [stepsForViz]);

  const sensorAsciiBlock = useMemo(() => {
    const palette = " .'`^,;:-+*=%#@";
    const MW = GRID_COLS;
    const rows: string[] = [];
    for (let r = 0; r < GRID_ROWS; r++) {
      let line = "";
      for (let c = 0; c < GRID_COLS; c++) {
        const i = r * GRID_COLS + c;
        const o = sensorOpacities[i] ?? 0.1;
        const t = Math.min(
          palette.length - 1,
          Math.max(0, Math.floor(o * palette.length)),
        );
        line += palette[t]!;
      }
      rows.push("|" + line + "|");
    }
    const tok = String(stepsForViz.length);
    const head = ` MEM ${tok}/${GRID_TOTAL} · spread map `.slice(0, MW).padEnd(MW);
    const rim = "+" + "=".repeat(MW) + "+";
    const sub = "+" + "-".repeat(MW) + "+";
    const footInner = " .'`^,;:-+*=#@ dim··bright··spread ".slice(0, MW).padEnd(MW);
    return [rim, "|" + head + "|", sub, ...rows, sub, "|" + footInner + "|", rim].join("\n");
  }, [sensorOpacities, stepsForViz.length]);

  const hasLogprobSteps = stepsForViz.length > 0;

  const telemStrip = useMemo(() => {
    const W = STRIP_IN;
    const p1pct = hasLogprobSteps ? `${(avgTop1 * 100).toFixed(1)}%` : "--%";
    const hSpread = hasLogprobSteps ? avgEntropy.toFixed(2) : "--.--";
    const toks = hasLogprobSteps ? String(stepsForViz.length) : "--";
    const cap = 2.5;
    const suffix1 = ` ${p1pct} sure`;
    const suffix2 = ` ${hSpread} spread`;
    const barW1 = Math.max(10, W - "|CF|".length - suffix1.length);
    const barW2 = Math.max(10, W - "|SP|".length - suffix2.length);

    const p1n = hasLogprobSteps ? Math.round(avgTop1 * barW1) : 0;
    const p1f = Math.max(0, Math.min(barW1, p1n));
    const p1rail = "#".repeat(p1f) + ".".repeat(barW1 - p1f);

    const hn = hasLogprobSteps ? Math.round((avgEntropy / cap) * barW2) : 0;
    const hf = Math.max(0, Math.min(barW2, hn));
    const hrail = "=".repeat(hf) + "-".repeat(barW2 - hf);

    const L = (s: string) => stripLine(s, W);
    const B = (ch: string) => stripBorder(ch, W);

    const wave = "~^".repeat(Math.ceil(W / 2)).slice(0, W);

    return [
      B("="),
      L(wave),
      L(
        " LIVE TELEMETRY · see locked-in vs hesitant reads while the model streams ",
      ),
      L(
        " Upper panels fill from the reply; model picker & message box below. ",
      ),
      B("-"),
      L(` HERO ${heroDisplay}  tok ${toks}`),
      L(`|CF|${p1rail}${suffix1}`),
      L(`|SP|${hrail}${suffix2}`),
      B("="),
    ].join("\n");
  }, [avgEntropy, avgTop1, hasLogprobSteps, heroDisplay, stepsForViz.length]);

  const selectedStep = stepsForViz[selectedIdx] ?? null;
  const altRows = useMemo(() => {
    if (!selectedStep) return [];
    const merged = [
      ...(selectedStep.top_logprobs ?? []),
      { token: selectedStep.token, logprob: selectedStep.logprob },
    ];
    const m = new Map<string, number>();
    for (const e of merged) {
      m.set(e.token, Math.max(m.get(e.token) ?? -Infinity, e.logprob));
    }
    return [...m.entries()]
      .map(([token, logprob]) => ({ token, logprob }))
      .sort((a, b) => b.logprob - a.logprob)
      .slice(0, 14);
  }, [selectedStep]);

  const vol = useMemo(() => {
    const e = meanEntropy(stepsForViz);
    const d = meanAbsDeltaLogprob(stepsForViz);
    const m = meanChosenMass(stepsForViz);
    return {
      entropyBar: sparkPct(e, 2.5),
      deltaBar: sparkPct(d, 2),
      massBar: sparkPct(m, 1),
    };
  }, [stepsForViz]);

  const hubAscii = useMemo(() => {
    const CW = 22;
    const cell = (s: string) => s.replace(/\r?\n/g, " ").slice(0, CW).padEnd(CW);
    const tri = (a: string, b: string, c: string) => `| ${cell(a)} | ${cell(b)} | ${cell(c)} |`;
    const sep = (ch: string) => `+${ch.repeat(24)}+${ch.repeat(24)}+${ch.repeat(24)}+`;
    const rail = (pct: number, n: number) => {
      const f = Math.max(0, Math.min(n, Math.round((pct / 100) * n)));
      return "#".repeat(f) + ".".repeat(n - f);
    };

    const candLines: string[] = [];
    if (!altRows.length) {
      candLines.push("(pick trace tok)");
    } else {
      for (const r of altRows.slice(0, 8)) {
        const tk = JSON.stringify(r.token).replace(/\s+/g, " ").slice(0, 11);
        const pct = `${(logprobToLinear(r.logprob) * 100).toFixed(0)}%`;
        candLines.push(`${tk} ${pct}`.slice(0, CW));
      }
    }

    const mdl = lastModel || model || defaultModel;
    const sessLines = [
      `MDL ${mdl.replace(/^[^/]+\//, "").slice(0, 17)}`,
      `LAT ${latencyMs != null ? latencyMs : "--"}ms`.slice(0, CW),
      `TOK ${stepsForViz.length} K=${TOP_LOGPROBS}`.slice(0, CW),
      `TMP ${effectiveTemperature.toFixed(2)}${hyperMode ? "H" : "_"}`.slice(0, CW),
    ];

    const rw = 6;
    const volLines = [
      `ENT ${rail(vol.entropyBar, rw)} ${vol.entropyBar}%`.slice(0, CW),
      `DLP ${rail(vol.deltaBar, rw)} ${vol.deltaBar}%`.slice(0, CW),
      `M1  ${rail(vol.massBar, rw)} ${vol.massBar}%`.slice(0, CW),
    ];

    const nRow = Math.max(candLines.length, sessLines.length, volLines.length);
    const lines: string[] = [];
    lines.push(sep("="));
    lines.push(tri(`[~] STEP ${selectedIdx}`, "MODEL · RUN", "LEVEL BARS"));
    lines.push(sep("-"));
    for (let i = 0; i < nRow; i++) {
      lines.push(tri(candLines[i] ?? "", sessLines[i] ?? "", volLines[i] ?? ""));
    }
    lines.push(sep("="));
    return lines.join("\n");
  }, [
    altRows,
    selectedIdx,
    lastModel,
    model,
    defaultModel,
    latencyMs,
    effectiveTemperature,
    hyperMode,
    stepsForViz.length,
    vol,
  ]);

  const darkHeadAscii = useMemo(() => {
    const proto =
      (model || defaultModel) + ` · TOP_${TOP_LOGPROBS}` + (hyperMode ? " · HYPER" : "");
    return [
      asciiCapTop("OPENROUTER LINK"),
      asciiRow(`LOGPROB GUI · ${proto}`),
      asciiRow(`LOCAL · ${clock}`),
      asciiCapBot(),
    ].join("\n");
  }, [clock, defaultModel, hyperMode, model]);

  const stopGeneration = useCallback(() => {
    completionAbortRef.current?.abort();
  }, []);

  const send = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    const prevMessages = messagesRef.current;
    setInput("");
    setErrorBanner(null);
    setLatencyMs(null);

    const hadAssistant = prevMessages.some((m) => m.role === "assistant");
    const nextMessages: ChatMessage[] = [...prevMessages, { role: "user", content: trimmed }];
    setMessages(nextMessages);
    setLive(null);
    streamPartialRef.current = { text: "", steps: [] };
    setLoading(true);
    setAgitated(true);
    const ac = new AbortController();
    completionAbortRef.current = ac;
    pushTrace(newTrace("REQ_START"));
    if (hadAssistant) pushTrace(newTrace("REGEN"));

    const t0 = performance.now();
    let sawSignal = false;

    const applyUserStop = (p: { text: string; steps: TokenLogprobStep[] }) => {
      pushTrace(newTrace("ABORT", "stream cancelled"));
      setLive(null);
      const latencyRound = Math.round(performance.now() - t0);
      setLatencyMs(latencyRound);
      if ((p.text && p.text.trim()) || p.steps.length > 0) {
        setLastModel(model || defaultModel);
        setLastCompletion({
          text: p.text,
          steps: p.steps,
          model: model || defaultModel,
        });
        setMessages([
          ...nextMessages,
          { role: "assistant", content: p.text.trim() ? p.text : "(stopped)" },
        ]);
        if (p.steps.length > 0) {
          pushTrace(newTrace("TOK_BATCH", `${p.steps.length} tokens (partial)`));
        }
      } else {
        setMessages(nextMessages);
      }
      streamPartialRef.current = { text: "", steps: [] };
      pushTrace(newTrace("STREAM_DONE"));
    };

    try {
      const res = await fetch("/api/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ac.signal,
        body: JSON.stringify({
          messages: nextMessages,
          model: model || undefined,
          stream: true,
          temperature: effectiveTemperature,
          top_logprobs: TOP_LOGPROBS,
        }),
      });

      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        const err = j as { error?: string; detail?: string; retryAfter?: string };
        const msg = [err.error, err.detail, err.retryAfter ? `retry after ${err.retryAfter}s` : ""]
          .filter(Boolean)
          .join(" · ");
        throw new Error(msg || res.statusText);
      }
      if (!res.body) throw new Error("No response body");

      const acc = await consumeOpenRouterSse(
        res.body,
        (partial) => {
          streamPartialRef.current = {
            text: partial.text,
            steps: partial.steps,
          };
          if (
            !sawSignal &&
            (partial.text.length > 0 || partial.steps.length > 0)
          ) {
            sawSignal = true;
            setAgitated(false);
            pushTrace(newTrace("FIRST_CHUNK"));
          }
          setLive({ text: partial.text, steps: partial.steps });
        },
        ac.signal,
      );

      if (ac.signal.aborted) {
        applyUserStop({ text: acc.text, steps: acc.steps });
        return;
      }

      const latencyRound = Math.round(performance.now() - t0);
      setLatencyMs(latencyRound);
      setLastModel(model || defaultModel);
      setLastCompletion({
        text: acc.text,
        steps: acc.steps,
        model: model || defaultModel,
      });
      setMessages([...nextMessages, { role: "assistant", content: acc.text }]);
      setLive(null);

      if (acc.steps.length === 0) {
        pushTrace(newTrace("NO_STEP_DETAIL", "stream had no per-step breakdown"));
      } else {
        pushTrace(newTrace("TOK_BATCH", `${acc.steps.length} tokens`));
      }
      pushTrace(newTrace("STREAM_DONE"));
    } catch (e) {
      const aborted =
        (typeof DOMException !== "undefined" &&
          e instanceof DOMException &&
          e.name === "AbortError") ||
        (e instanceof Error && e.name === "AbortError");
      if (aborted) {
        applyUserStop(streamPartialRef.current);
      } else {
        const msg = e instanceof Error ? e.message : String(e);
        setErrorBanner(msg);
        pushTrace(newTrace("ERR", msg));
        setMessages(prevMessages);
      }
    } finally {
      completionAbortRef.current = null;
      setAgitated(false);
      setLoading(false);
    }
  };

  return (
    <div className="shell ascii-shell">
      <div className={`zone-dark ascii-zone-upper crt-zone ${agitated ? "agitated" : ""}`}>
        <div className="ascii-dark-head">
          <pre className="ascii-pre ascii-pre--mast" aria-hidden>
            {DARK_TITLE_BANNER}
          </pre>
          <pre
            className="ascii-pre ascii-pre--dark ascii-pre--headbox"
            aria-label={`Logprob GUI; model ${model || defaultModel}; ${clock}`}
          >
            {darkHeadAscii}
          </pre>
        </div>

        <div className="dark-vis-stack ascii-dark-vis">
          <div className="ascii-pre-wrap ascii-pre-wrap--ribbon">
            <pre className="ascii-pre ascii-pre--dark" aria-hidden>
              {telemStrip}
            </pre>
          </div>
          <div className="ascii-pre-wrap ascii-pre-wrap--hub">
            <pre className="ascii-pre ascii-pre--dark" aria-hidden>
              {hubAscii}
            </pre>
          </div>
          <div className="ascii-pre-wrap ascii-pre-wrap--sensor">
            <pre
              className="ascii-pre ascii-pre--dark"
              role="img"
              aria-label={`Spread map ${GRID_COLS} by ${GRID_ROWS}; ${stepsForViz.length} steps`}
            >
              {sensorAsciiBlock}
            </pre>
          </div>
        </div>
      </div>

      <pre className="ascii-pre ascii-pre--split" aria-hidden>
        {ZONE_SPLIT_LINE}
      </pre>

      <div className="zone-light ascii-zone-lower crt-zone">
        <div className="ascii-banner-controls">
          <pre className="ascii-pre ascii-pre--light-banner" aria-hidden>
            {LIGHT_ZONE_BANNER}
          </pre>
          <div className="ascii-banner-param-cluster">
            <label className="ascii-field">
              <span className="sr-only">Model</span>
              <span className="ascii-select-wrap">
                <select
                  className="ascii-select"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  {modelOptions.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </span>
            </label>
            <label className={`ascii-field ${hyperMode ? "ascii-field--dim" : ""}`}>
              <span className="sr-only">Temperature</span>
              <span className="ascii-field-hint" aria-hidden>
                T={hyperMode ? "2.00" : temperature.toFixed(2)}
              </span>
              <input
                type="range"
                min={0}
                max={TEMP_SLIDER_MAX}
                step={0.05}
                value={Math.min(temperature, TEMP_SLIDER_MAX)}
                disabled={hyperMode}
                onChange={(e) =>
                  setTemperature(
                    Math.min(TEMP_SLIDER_MAX, Math.max(0, Number(e.target.value))),
                  )
                }
              />
            </label>
            <button
              className={`ascii-btn ${hyperMode ? "ascii-btn--hot" : "ascii-btn--ghost"}`}
              type="button"
              title="Temperature fixed at 2 (max chaos)"
              aria-pressed={hyperMode}
              onClick={() => setHyperMode((h) => !h)}
            >
              HYPER
            </button>
          </div>
        </div>

        {errorBanner ? (
          <pre className="ascii-pre ascii-pre--error" role="alert">
            {[asciiCapTop("ERR"), asciiRow(errorBanner), asciiCapBot()].join("\n")}
          </pre>
        ) : null}

        <div className="ascii-main-grid">
          <div className="ascii-col ascii-col--primary">
            <div className="ascii-panel">
              <pre className="ascii-pre ascii-pre--cap" aria-hidden>
                {stepsForViz.length > 0 || textForViz
                  ? [asciiCapTop("TRACE · click token"), asciiCapMid()].join("\n")
                  : asciiEmptyTrace()}
              </pre>
              {stepsForViz.length === 0 && !textForViz ? (
                <span className="sr-only">No step detail yet.</span>
              ) : null}
              {(stepsForViz.length > 0 || textForViz) && (
                <div className="ascii-panel-body ascii-panel-body--trace">
                  {stepsForViz.length > 0 ? (
                    <div className="ascii-flow">
                      {stepsForViz.map((st, i) => {
                        const ent = entropyFromStep(st);
                        const p1 = top1LinearProb(st);
                        const solid = 0.42 + 0.58 * p1;
                        const candsTitle = spectrumTokens(st);
                        const cands = competitiveSpectrumTokens(st);
                        const selected = i === selectedIdx;
                        const ambiguous =
                          cands.length > 1 &&
                          p1 < TRACE_GLITCH_MAX_P1 &&
                          ent >= TRACE_GLITCH_MIN_ENT;
                        const slotLen = morphSlotLen(st.token, cands);
                        const morphActive = ambiguous && !selected;
                        return (
                          <span
                            key={i}
                            className={
                              "token-span" +
                              (ambiguous ? " token-span--slot" : "") +
                              (selected ? " selected" : "") +
                              (morphActive ? " token-span--morph" : "")
                            }
                            style={{
                              opacity: solid,
                              ...(ambiguous ? { minWidth: `${slotLen}ch` } : {}),
                            }}
                            title={
                              `picked ${JSON.stringify(st.token)} · spread ${ent.toFixed(2)} · leaning ${(100 * p1).toFixed(1)}% · runners ` +
                              candsTitle
                                .slice(0, 10)
                                .map((t) => JSON.stringify(t))
                                .join(" ")
                            }
                            onClick={() => setSelectedIdx(i)}
                          >
                            {morphActive ? (
                              <>
                                <span className="sr-only">{st.token}</span>
                                <TraceMorphTokenCols
                                  cands={cands}
                                  morphPhase={morphPhase}
                                  slotLen={slotLen}
                                />
                              </>
                            ) : (
                              st.token
                            )}
                          </span>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="ascii-flow">{textForViz}</div>
                  )}
                  <pre className="ascii-pre ascii-pre--cap ascii-pre--cap-tail" aria-hidden>
                    {asciiCapBot()}
                  </pre>
                </div>
              )}
            </div>

            <div className="ascii-panel">
              <pre className="ascii-pre ascii-pre--cap" aria-hidden>
                {[asciiCapTop("send"), asciiCapMid()].join("\n")}
              </pre>
              <div className="ascii-panel-body ascii-panel-body--tx">
                <pre className="ascii-pre ascii-pre--io-gutter" aria-hidden>
                  {`>>>`} Enter sends · Shift+Enter newline {`<<<`}
                </pre>
                <div className="ascii-tx-frame">
                  <pre className="ascii-pre ascii-pre--io-edge" aria-hidden>
                    {`+${"─".repeat(70)}+`}
                  </pre>
                  <textarea
                    className="ascii-textarea"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="> Type your message..."
                    aria-label="User message"
                    spellCheck={false}
                    onKeyDown={(e) => {
                      if (e.key !== "Enter") return;
                      if (e.nativeEvent.isComposing) return;
                      if (e.shiftKey) return;
                      e.preventDefault();
                      void send();
                    }}
                  />
                  <pre className="ascii-pre ascii-pre--io-edge" aria-hidden>
                    {`+${"─".repeat(70)}+`}
                  </pre>
                </div>
                <div className="ascii-tx-actions">
                  <button
                    className="ascii-btn ascii-btn--solid"
                    type="button"
                    disabled={loading}
                    onClick={() => void send()}
                  >
                    send
                  </button>
                  <button
                    className="ascii-btn ascii-btn--ghost"
                    type="button"
                    disabled={!loading}
                    onClick={stopGeneration}
                  >
                    STOP
                  </button>
                </div>
              </div>
              <pre className="ascii-pre ascii-pre--cap ascii-pre--cap-tail" aria-hidden>
                {asciiCapBot()}
              </pre>
            </div>
          </div>

          <div className="ascii-col ascii-col--side">
            <div className="ascii-panel ascii-panel--grow">
              <pre className="ascii-pre ascii-pre--cap" aria-hidden>
                {[asciiCapTop("EVENT LOG"), asciiCapMid()].join("\n")}
              </pre>
              <div className="ascii-panel-body ascii-panel-body--events">
                <ul className="ascii-event-list">
                  {traces.map((t) => (
                    <li key={t.id} className="ascii-event-row">
                      <span className="ascii-event-ts">
                        {new Date(t.ts).toISOString().slice(11, 23)}
                      </span>
                      <span className="ascii-event-kind">
                        {t.kind}
                        {t.detail ? ` · ${t.detail}` : ""}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              <pre className="ascii-pre ascii-pre--cap ascii-pre--cap-tail" aria-hidden>
                {asciiCapBot()}
              </pre>
            </div>

            <div className="ascii-panel ascii-panel--grow">
              <pre className="ascii-pre ascii-pre--cap" aria-hidden>
                {!messages.length && !live
                  ? asciiEmptyDialog()
                  : [asciiCapTop("DIALOG"), asciiCapMid()].join("\n")}
              </pre>
              {!messages.length && !live ? (
                <span className="sr-only">
                  Waiting for a message. Send one to fill the upper panels and charts.
                </span>
              ) : null}
              {(messages.length > 0 || live) && (
                <>
                  <div className="ascii-panel-body ascii-panel-body--dialog">
                    {messages.map((m, i) => (
                      <div key={i} className={`ascii-msg ascii-msg--${m.role}`}>
                        <pre className="ascii-msg-tag" aria-hidden>{`[${m.role}]`}</pre>
                        <div className="ascii-msg-text">{m.content}</div>
                      </div>
                    ))}
                    {live ? (
                      <div className="ascii-msg ascii-msg--assistant">
                        <pre className="ascii-msg-tag" aria-hidden>
                          [assistant · stream]
                        </pre>
                        <div className="ascii-msg-text ascii-msg-text--live">{live.text}</div>
                      </div>
                    ) : null}
                  </div>
                  <pre className="ascii-pre ascii-pre--cap ascii-pre--cap-tail" aria-hidden>
                    {asciiCapBot()}
                  </pre>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
