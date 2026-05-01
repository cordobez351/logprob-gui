export type ChatRole = "system" | "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

/** One alternative at a generation step (OpenAI-compatible). */
export type TopLogprobEntry = {
  token: string;
  logprob: number;
  bytes?: number[] | null;
};

/** One output token position with chosen + top-k. */
export type TokenLogprobStep = {
  token: string;
  logprob: number;
  top_logprobs: TopLogprobEntry[] | null;
};

export type TraceEventKind =
  | "REQ_START"
  | "FIRST_CHUNK"
  | "STREAM_DONE"
  | "TOK_BATCH"
  | "NO_STEP_DETAIL"
  | "ERR"
  | "REGEN"
  | "ABORT";

export type TraceEvent = {
  id: string;
  ts: number;
  kind: TraceEventKind;
  detail?: string;
};

export type CompletionSnapshot = {
  id: string;
  createdAt: number;
  label: string;
  model: string;
  messages: ChatMessage[];
  assistantText: string;
  steps: TokenLogprobStep[];
  meanTop1Percent: number | null;
};

export type OpenRouterChatChunk = {
  choices?: Array<{
    delta?: { content?: string | null; role?: string | null };
    finish_reason?: string | null;
    logprobs?: { content?: TokenLogprobStep[] | null } | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
};

export type OpenRouterChatResponse = {
  choices?: Array<{
    message?: { role?: string; content?: string | null };
    finish_reason?: string | null;
    logprobs?: { content?: TokenLogprobStep[] | null } | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  model?: string;
  error?: { message?: string };
};
