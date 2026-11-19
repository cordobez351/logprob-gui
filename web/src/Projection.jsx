import { useEffect, useState } from 'react';

import { alternatives, competitiveTokens, entropyFromStep, top1LinearProb } from './logprobs.js';

// A token flickers between its real alternates only when there was real spread.
const MAX_P1 = 0.9995;
const MIN_ENT = 0.01;
const MORPH_MAX_COLS = 32;
const MORPH_CYCLE_MS = 5200;
const GOLDEN = 0.6180339887498949; // spaces the per-token phases so they never line up
const GLITCH = '·░▒▓*:;';

const pad = (s, n) => (s.length >= n ? s.slice(0, n) : s + ' '.repeat(n - s.length));
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const shimmer = (phase, i) => 0.94 + 0.06 * Math.sin(2 * Math.PI * (phase * 2.7 + i * 0.53));

// Column-by-column crossfade between two candidate tokens (from the first version).
function Morph({ cands, phase, len }) {
  const n = cands.length;
  const cont = Math.min(n - 1e-9, phase * n);
  const k = Math.floor(cont) % n;
  const lp = cont - Math.floor(cont);
  const from = pad(cands[k] ?? '', len), to = pad(cands[(k + 1) % n] ?? '', len);
  const mid = (len - 1) / 2, maxDist = Math.max(mid, 1e-6);
  return Array.from({ length: len }, (_, i) => {
    const a = from[i], b = to[i];
    if (a === b) return <span key={i} className="col"><span style={{ opacity: shimmer(phase, i) }}>{a}</span></span>;
    const lag = (Math.abs(i - mid) / maxDist) * 0.5;
    const blend = smooth(0.06, 0.94, (lp - lag) / Math.max(0.12, 1 - lag * 0.85));
    const rise = Math.sin(Math.PI * blend);
    const glitch = rise > 0.38 && rise < 0.72 && Math.abs(blend - 0.5) < 0.18;
    const sh = shimmer(phase, i);
    return (
      <span key={i} className="col">
        <span style={{ opacity: (1 - blend) * (0.45 + 0.55 * (1 - rise)) * sh, transform: `scaleY(${0.28 + 0.72 * (1 - blend)})` }}>{a}</span>
        {glitch && <span className="col-glitch" style={{ opacity: Math.sin(Math.PI * ((blend - 0.28) / 0.44)) * 0.88 }}>{GLITCH[(i + k * 7 + Math.floor(lp * 60)) % GLITCH.length]}</span>}
        <span style={{ opacity: Math.min(1, blend + rise * 0.35) * sh, transform: `scaleY(${0.22 + 0.78 * blend})` }}>{b}</span>
      </span>
    );
  });
}

// The reply on the screen. Sure tokens come out sharp; unsure ones stay soft and flicker.
export function Projection({ steps, text, selected, onSelect, branchAt }) {
  const [phase, setPhase] = useState(0);
  useEffect(() => {
    if (!steps.length) return undefined;
    let raf = 0;
    const tick = () => { setPhase((Date.now() % MORPH_CYCLE_MS) / MORPH_CYCLE_MS); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [steps.length > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!steps.length) return <div className="reply">{text}</div>;
  return (
    <div className="reply">
      {steps.map((st, i) => {
        const p1 = top1LinearProb(st);
        const ent = entropyFromStep(st);
        const cands = competitiveTokens(st);
        const ambiguous = cands.length > 1 && p1 < MAX_P1 && ent >= MIN_ENT;
        const len = Math.max(1, Math.min(MORPH_MAX_COLS, Math.max(st.token.length, ...cands.map((c) => c.length))));
        const sel = i === selected;
        const morph = ambiguous && !sel;
        return (
          <span
            key={i}
            className={`tok${ambiguous ? ' unsure' : ''}${sel ? ' sel' : ''}${i === branchAt ? ' fork' : ''}`}
            style={{
              opacity: 0.42 + 0.58 * p1,
              filter: sel ? 'none' : `blur(${((1 - p1) * 1.4).toFixed(2)}px)`,
              ...(ambiguous ? { minWidth: `${len}ch` } : {}),
            }}
            onClick={() => onSelect(i)}
          >
            {morph ? <><span className="sr">{st.token}</span><Morph cands={cands} phase={(phase + i * GOLDEN) % 1} len={len} /></> : st.token}
          </span>
        );
      })}
    </div>
  );
}

const show = (tok) => JSON.stringify(tok).slice(1, -1).replace(/^ /, '␣').replace(/ $/, '␣') || '∅';

// The futures at one step: a fan of lines from the past into each candidate.
// Tapping a candidate the model did not pick opens that future (`onPick`).
export function Worlds({ step, onPick }) {
  const alts = alternatives(step, 8);
  const rowH = 30, top = 16, h = top * 2 + rowH * (alts.length - 1), ox = 24, oy = h / 2, bx = 210;
  return (
    <svg className="worlds" viewBox={`0 0 640 ${h}`} role="img" aria-label={alts.map((a) => `${a.token} ${Math.round(a.p * 100)}%`).join(', ')}>
      <circle cx={ox} cy={oy} r="4" className="worlds-origin" />
      {alts.map((a, i) => {
        const y = top + i * rowH;
        return (
          <g
            key={a.token + i}
            className={a.chosen ? 'world chosen' : onPick ? 'world open' : 'world'}
            onClick={!a.chosen && onPick ? () => onPick(a.token, Math.log(a.p)) : undefined}
          >
            <rect x={0} y={y - rowH / 2} width={640} height={rowH} className="world-hit" />
            <path
              d={`M${ox},${oy} C${ox + 90},${oy} ${bx - 90},${y} ${bx},${y}`}
              strokeWidth={0.8 + a.p * 5}
              strokeOpacity={0.18 + a.p * 0.82}
            />
            <circle cx={bx} cy={y} r={2 + a.p * 3} />
            <text x={bx + 14} y={y + 4} className="world-tok">{show(a.token)}</text>
            <rect x={430} y={y - 3} width={160} height={6} rx={3} className="world-track" />
            <rect x={430} y={y - 3} width={Math.max(2, a.p * 160)} height={6} rx={3} className="world-fill" />
            <text x={632} y={y + 4} textAnchor="end" className="world-pct">{(a.p * 100).toFixed(a.p < 0.1 ? 1 : 0)}%</text>
          </g>
        );
      })}
    </svg>
  );
}

// Every token as a bar: tall = sure, short = the model hesitated.
export function Timeline({ steps, selected, onSelect, label }) {
  return (
    <div className="timeline" role="listbox" aria-label={label}>
      {steps.map((st, i) => {
        const p1 = top1LinearProb(st);
        return (
          <button
            key={i}
            role="option"
            aria-selected={i === selected}
            className={i === selected ? 'tick sel' : 'tick'}
            style={{ '--h': `${12 + p1 * 88}%` }}
            title={`${show(st.token)} · ${Math.round(p1 * 100)}%`}
            onClick={() => onSelect(i)}
          />
        );
      })}
    </div>
  );
}
