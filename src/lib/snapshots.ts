import type { CompletionSnapshot } from "./types";

const STORAGE_KEY = "logprob-gui-snapshots";

export function loadSnapshots(): CompletionSnapshot[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isSnapshot);
  } catch {
    return [];
  }
}

function isSnapshot(x: unknown): x is CompletionSnapshot {
  if (!x || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.id === "string" &&
    typeof o.createdAt === "number" &&
    typeof o.label === "string" &&
    typeof o.model === "string" &&
    typeof o.assistantText === "string" &&
    Array.isArray(o.messages) &&
    Array.isArray(o.steps)
  );
}

export function saveSnapshot(s: CompletionSnapshot): void {
  const cur = loadSnapshots();
  const next = [s, ...cur].slice(0, 24);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
}

export function deleteSnapshot(id: string): void {
  const cur = loadSnapshots().filter((s) => s.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cur));
}
