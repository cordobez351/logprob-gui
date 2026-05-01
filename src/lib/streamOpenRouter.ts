import type { OpenRouterChatChunk, TokenLogprobStep } from "./types";

export type StreamAccum = {
  text: string;
  steps: TokenLogprobStep[];
  usage?: OpenRouterChatChunk["usage"];
  finishReason?: string | null;
};

function appendDeltaText(acc: StreamAccum, chunk: string | null | undefined) {
  if (chunk) acc.text += chunk;
}

export async function consumeOpenRouterSse(
  body: ReadableStream<Uint8Array>,
  onChunk: (partial: StreamAccum) => void,
  signal?: AbortSignal,
): Promise<StreamAccum> {
  const acc: StreamAccum = { text: "", steps: [] };
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flushBlock = (block: string) => {
    for (const line of block.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (data === "[DONE]") continue;
      let json: unknown;
      try {
        json = JSON.parse(data);
      } catch {
        continue;
      }
      const obj = json as OpenRouterChatChunk & { error?: { message?: string } };
      if (obj.error?.message) {
        throw new Error(obj.error.message);
      }
      const choice = obj.choices?.[0];
      if (choice?.delta?.content) {
        appendDeltaText(acc, choice.delta.content);
      }
      const lpContent = choice?.logprobs?.content;
      if (Array.isArray(lpContent) && lpContent.length > 0) {
        for (const step of lpContent) acc.steps.push(step);
      }
      if (choice?.finish_reason) acc.finishReason = choice.finish_reason;
      if (obj.usage) acc.usage = obj.usage;
      onChunk({ ...acc, steps: [...acc.steps], text: acc.text });
    }
  };

  while (true) {
    let chunk: ReadableStreamReadResult<Uint8Array>;
    try {
      chunk = await reader.read();
    } catch (err) {
      if (signal?.aborted) {
        try {
          await reader.cancel();
        } catch {
          /* ignore */
        }
        return acc;
      }
      throw err;
    }
    const { done, value } = chunk;
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const p of parts) {
      if (p.trim()) flushBlock(p);
    }
  }
  if (buffer.trim()) flushBlock(buffer);
  return acc;
}
