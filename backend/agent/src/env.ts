import { config } from 'dotenv';
import { resolve } from 'node:path';

// The single .env at the assignment root. Provider keys are read HERE and nowhere else.
config({ path: resolve(process.cwd(), '../../.env') });
config({ path: resolve(process.cwd(), '.env') });

const num = (v: string | undefined, fallback: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};
export const env = {
  port: num(process.env.PORT_AGENT ?? process.env.PORT, 8000),
  mongoUri: process.env.MONGODB_URI ?? '',
  mongoDb: process.env.MONGODB_DB ?? 'lumina',
  vectorBackend: (process.env.VECTOR_BACKEND ?? 'atlas-vector-search') as
    | 'atlas-vector-search'
    | 'mongo-cosine-scan',

  llmProvider: process.env.LLM_PROVIDER ?? 'anthropic',
  llmModel: process.env.LLM_MODEL ?? 'claude-sonnet-5',

  searchProvider: (process.env.SEARCH_PROVIDER ?? 'tavily') as 'tavily' | 'serpapi',
  searchCacheTtlSeconds: num(process.env.SEARCH_CACHE_TTL_SECONDS, 21600),

  embeddingModel: process.env.EMBEDDING_MODEL ?? 'text-embedding-3-small',

  // Chunking: small enough that a citation points at one passage, big enough to carry its
  // context. Chunks never cross a PDF page, so a page locator is always exact.
  chunkChars: num(process.env.CHUNK_CHARS, 1000),
  chunkOverlapChars: num(process.env.CHUNK_OVERLAP_CHARS, 150),
  // How long the read-your-write probe waits for Atlas Search to catch up before the job is retried.
  probeTimeoutSec: num(process.env.PROBE_TIMEOUT_SEC, 90),

  // Hybrid retrieval. Declared here, not buried in code, so a recall number can be traced
  // to the settings that produced it.
  docsTopK: num(process.env.DOCS_TOP_K, 6), // chunks handed to the answer (and listed as sources)
  retrievalCandidates: num(process.env.RETRIEVAL_CANDIDATES, 20), // per retriever, before fusion
  rrfK: num(process.env.RRF_K, 60), // reciprocal rank fusion constant: score = Σ 1 / (k + rank)
  // mode=auto uses the Space when its best vector match scores at least this (Atlas cosine
  // scores are (1 + cos) / 2, so 0.5 is "unrelated" and 1.0 is identical). Calibrated on the
  // gold corpus: its 39 questions scored 0.64–0.75 (median 0.75), unrelated questions 0.56–0.66.
  autoDocMinScore: num(process.env.AUTO_DOC_MIN_SCORE, 0.66),

  // Deep search is the expensive gear, so its limits are configuration, not code.
  deepSubQuestionsMin: num(process.env.DEEP_SUB_QUESTIONS_MIN, 3),
  deepSubQuestionsMax: num(process.env.DEEP_SUB_QUESTIONS_MAX, 6),
  deepDailyCap: num(process.env.DEEP_DAILY_CAP, 5),

  // The hard caps from AGENTS.md. Raising these to make a gate pass is the failure mode
  // the caps exist to catch. Two gears, two envelopes.
  maxToolCalls: num(process.env.MAX_TOOL_CALLS, 8),
  maxWallClockSec: num(process.env.MAX_WALL_CLOCK_SEC, 90),
  maxToolCallsDeep: num(process.env.MAX_TOOL_CALLS_DEEP, 24),
  maxWallClockSecDeep: num(process.env.MAX_WALL_CLOCK_SEC_DEEP, 240),

  logLevel: process.env.LOG_LEVEL ?? 'info',
  /** Where the per-answer run logs land. quality/check.mjs reads this folder. */
  runsDir: resolve(process.cwd(), '../../runs')
} as const;

/** Never log or return these. /health names the model; it never echoes a key. */
export const secrets = {
  anthropic: process.env.ANTHROPIC_API_KEY ?? '',
  openai: process.env.OPENAI_API_KEY ?? '',
  tavily: process.env.TAVILY_API_KEY ?? '',
  serpapi: process.env.SERPAPI_API_KEY ?? ''
} as const;
