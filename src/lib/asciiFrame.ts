export const BOX_TW = 76;

export function asciiCapTop(title: string, tw = BOX_TW): string {
  const inner = tw - 2;
  let t = title;
  while (t.length > 0 && `══ ${t} `.length > inner) {
    t = t.slice(0, -1);
  }
  const prefix = `══ ${t} `;
  const fills = Math.max(0, inner - prefix.length);
  return `╔${prefix}${"═".repeat(fills)}╗`;
}

export function asciiCapBot(tw = BOX_TW): string {
  return `╚${"═".repeat(tw - 2)}╝`;
}

export function asciiCapMid(tw = BOX_TW): string {
  return `╠${"═".repeat(tw - 2)}╣`;
}

export function asciiRow(inner: string, tw = BOX_TW): string {
  const innerW = tw - 4;
  const t = inner.replace(/\r?\n/g, " ").slice(0, innerW).padEnd(innerW);
  return `║ ${t} ║`;
}

export function asciiEmptyTrace(): string {
  return [
    asciiCapTop("TRACE · click token"),
    asciiRow("· · nothing to show yet — send a message first · ·"),
    asciiCapBot(),
  ].join("\n");
}

export function asciiEmptyDialog(): string {
  return [
    asciiCapTop("DIALOG"),
    asciiRow("· · conversation shows here after you send · ·"),
    asciiCapBot(),
  ].join("\n");
}

export function asciiEmptyAnalysis(): string {
  return [
    asciiCapTop("2ND AI · idle"),
    asciiRow("· · appears once a run returns rich step detail · ·"),
    asciiCapBot(),
  ].join("\n");
}
