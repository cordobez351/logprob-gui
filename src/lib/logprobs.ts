import type { TokenLogprobStep } from "./types";

export function logprobToLinear(logprob: number): number {
  return Math.exp(logprob);
}

function mergedLogprobs(step: TokenLogprobStep): Map<string, number> {
  const m = new Map<string, number>();
  for (const e of step.top_logprobs ?? []) {
    m.set(e.token, Math.max(m.get(e.token) ?? -Infinity, e.logprob));
  }
  m.set(step.token, Math.max(m.get(step.token) ?? -Infinity, step.logprob));
  return m;
}

/** Shannon entropy (nats) over normalized distribution from merged top + chosen. */
export function entropyFromStep(step: TokenLogprobStep): number {
  const m = mergedLogprobs(step);
  const logps = [...m.values()];
  if (logps.length === 0) return 0;
  const maxL = Math.max(...logps);
  let sum = 0;
  const probs: number[] = [];
  for (const lp of logps) {
    const p = Math.exp(lp - maxL);
    probs.push(p);
    sum += p;
  }
  if (sum <= 0) return 0;
  let h = 0;
  for (const p of probs) {
    const q = p / sum;
    if (q > 0) h -= q * Math.log(q);
  }
  return h;
}

export function top1LinearProb(step: TokenLogprobStep): number {
  const m = mergedLogprobs(step);
  const logps = [...m.values()];
  if (logps.length === 0) return logprobToLinear(step.logprob);
  const maxL = Math.max(...logps);
  let sum = 0;
  for (const lp of logps) {
    sum += Math.exp(lp - maxL);
  }
  return Math.exp(step.logprob - maxL) / sum;
}

export function meanTop1Percent(steps: TokenLogprobStep[]): number | null {
  if (steps.length === 0) return null;
  let s = 0;
  for (const st of steps) s += top1LinearProb(st);
  return Math.round((100 * s) / steps.length);
}

export function meanEntropy(steps: TokenLogprobStep[]): number {
  if (steps.length === 0) return 0;
  let s = 0;
  for (const st of steps) s += entropyFromStep(st);
  return s / steps.length;
}

export function meanAbsDeltaLogprob(steps: TokenLogprobStep[]): number {
  if (steps.length < 2) return 0;
  let s = 0;
  for (let i = 1; i < steps.length; i++) {
    s += Math.abs(steps[i]!.logprob - steps[i - 1]!.logprob);
  }
  return s / (steps.length - 1);
}

export function meanChosenMass(steps: TokenLogprobStep[]): number {
  if (steps.length === 0) return 0;
  let s = 0;
  for (const st of steps) s += top1LinearProb(st);
  return s / steps.length;
}

export function parseStepsFromResponse(
  data: { choices?: Array<{ logprobs?: { content?: TokenLogprobStep[] | null } | null }> },
): TokenLogprobStep[] {
  const content = data.choices?.[0]?.logprobs?.content;
  return Array.isArray(content) ? content : [];
}

export function mergeStreamLogprobSteps(chunks: TokenLogprobStep[][]): TokenLogprobStep[] {
  const out: TokenLogprobStep[] = [];
  for (const arr of chunks) {
    for (const step of arr) out.push(step);
  }
  return out;
}

/** Candidate strings for one step, highest logprob first (for trace cycling). */
export function spectrumTokens(step: TokenLogprobStep): string[] {
  const m = mergedLogprobs(step);
  if (m.size === 0) return [step.token];
  return [...m.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([tok]) => tok);
}

/** Subset of spectrum within `deltaLogprob` nats of the peak — for glitch cycling among real alternates. */
export function competitiveSpectrumTokens(
  step: TokenLogprobStep,
  deltaLogprob = 5,
): string[] {
  const m = mergedLogprobs(step);
  if (m.size === 0) return [step.token];
  const maxL = Math.max(...m.values());
  return [...m.entries()]
    .filter(([, lp]) => lp >= maxL - deltaLogprob)
    .sort((a, b) => b[1] - a[1])
    .map(([tok]) => tok);
}
