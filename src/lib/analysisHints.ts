import { entropyFromStep } from "./logprobs";
import type { TokenLogprobStep } from "./types";

/** Token → max logprob at this step (chosen ∪ top_logprobs). */
function mergedEntries(step: TokenLogprobStep): [string, number][] {
  const m = new Map<string, number>();
  for (const e of step.top_logprobs ?? []) {
    m.set(e.token, Math.max(m.get(e.token) ?? -Infinity, e.logprob));
  }
  m.set(step.token, Math.max(m.get(step.token) ?? -Infinity, step.logprob));
  return [...m.entries()];
}

/**
 * Human-readable digest so the analyzer model can anchor on real hotspots
 * without re-scanning megabytes of JSON.
 */
export function formatAnalysisHints(steps: TokenLogprobStep[]): string {
  if (steps.length === 0) return "(no token steps)";

  const lines: string[] = [];

  const byH = steps
    .map((s, i) => ({ i, s, h: entropyFromStep(s) }))
    .sort((a, b) => b.h - a.h)
    .slice(0, 6);
  lines.push("Highest-entropy steps (where the distribution was widest):");
  for (const { i, s, h } of byH) {
    const preview = s.token.length > 40 ? `${s.token.slice(0, 40)}…` : s.token;
    lines.push(`  [${i}] H=${h.toFixed(2)} nats · ${JSON.stringify(preview)}`);
  }

  const races: { i: number; gap: number; chosen: string; runner: string }[] = [];
  for (let i = 0; i < steps.length; i++) {
    const ent = mergedEntries(steps[i]!).sort((a, b) => b[1] - a[1]);
    if (ent.length < 2) continue;
    const gap = ent[0]![1] - ent[1]![1];
    races.push({
      i,
      gap,
      chosen: ent[0]![0],
      runner: ent[1]![0],
    });
  }
  races.sort((a, b) => a.gap - b.gap);
  lines.push("");
  lines.push("Tightest top-2 races (smallest Δlogprob between rank-1 and rank-2 token):");
  for (const r of races.slice(0, 6)) {
    lines.push(
      `  [${r.i}] Δ=${r.gap.toFixed(3)} nats · #1 ${JSON.stringify(r.chosen)} vs #2 ${JSON.stringify(r.runner)}`,
    );
  }

  return lines.join("\n");
}
