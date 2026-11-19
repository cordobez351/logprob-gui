// Same math as src/lib/logprobs.ts of the first version.

/** Token -> best logprob at this step (chosen ∪ top_logprobs). */
function merged(step) {
  const m = new Map();
  for (const e of step.top_logprobs ?? []) m.set(e.token, Math.max(m.get(e.token) ?? -Infinity, e.logprob));
  m.set(step.token, Math.max(m.get(step.token) ?? -Infinity, step.logprob));
  return m;
}

/** Shannon entropy (nats) over the normalized merged distribution. */
export function entropyFromStep(step) {
  const logps = [...merged(step).values()];
  if (!logps.length) return 0;
  const maxL = Math.max(...logps);
  const probs = logps.map((lp) => Math.exp(lp - maxL));
  const sum = probs.reduce((a, b) => a + b, 0);
  if (sum <= 0) return 0;
  let h = 0;
  for (const p of probs) { const q = p / sum; if (q > 0) h -= q * Math.log(q); }
  return h;
}

/** Share of the merged distribution that the chosen token holds. */
export function top1LinearProb(step) {
  const logps = [...merged(step).values()];
  if (!logps.length) return Math.exp(step.logprob);
  const maxL = Math.max(...logps);
  const sum = logps.reduce((a, lp) => a + Math.exp(lp - maxL), 0);
  return Math.exp(step.logprob - maxL) / sum;
}

export function meanTop1(steps) {
  if (!steps.length) return null;
  return steps.reduce((a, s) => a + top1LinearProb(s), 0) / steps.length;
}

export function meanEntropy(steps) {
  if (!steps.length) return 0;
  return steps.reduce((a, s) => a + entropyFromStep(s), 0) / steps.length;
}

/** All candidates at a step, highest first, as { token, p } with p = exp(logprob). */
export function alternatives(step, max = 8) {
  return [...merged(step).entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, max)
    .map(([token, lp]) => ({ token, p: Math.exp(lp), chosen: token === step.token }));
}

/** Candidates within `delta` nats of the peak: the real alternates a token flickers between. */
export function competitiveTokens(step, delta = 5) {
  const m = merged(step);
  const maxL = Math.max(...m.values());
  return [...m.entries()].filter(([, lp]) => lp >= maxL - delta).sort((a, b) => b[1] - a[1]).map(([t]) => t);
}
