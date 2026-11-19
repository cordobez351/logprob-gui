// OpenRouter SSE reader (same as src/lib/streamOpenRouter.ts of the first version).
export async function consumeOpenRouterSse(body, onChunk, signal) {
  const acc = { text: '', steps: [] };
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const flushBlock = (block) => {
    for (const line of block.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const data = trimmed.slice(5).trim();
      if (data === '[DONE]') continue;
      let obj;
      try { obj = JSON.parse(data); } catch { continue; }
      if (obj.error?.message) throw new Error(obj.error.message);
      const choice = obj.choices?.[0];
      if (choice?.delta?.content) acc.text += choice.delta.content;
      const lp = choice?.logprobs?.content;
      if (Array.isArray(lp)) for (const step of lp) acc.steps.push(step);
      if (choice?.finish_reason) acc.finishReason = choice.finish_reason;
      if (obj.usage) acc.usage = obj.usage;
      onChunk({ ...acc, steps: [...acc.steps] });
    }
  };

  while (true) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (err) {
      if (signal?.aborted) {
        try { await reader.cancel(); } catch { /* ignore */ }
        return acc;
      }
      throw err;
    }
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';
    for (const p of parts) if (p.trim()) flushBlock(p);
  }
  if (buffer.trim()) flushBlock(buffer);
  return acc;
}
