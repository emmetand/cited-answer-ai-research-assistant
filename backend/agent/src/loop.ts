import Anthropic from '@anthropic-ai/sdk';
import {
  newId,
  unresolvedCitations,
  type AskMode,
  type AskTool,
  type DoneEvent,
  type Locator,
  type SubQuestion,
  type RunLog,
  type Source,
  type Terminated
} from '@lumina/contract';
import { env } from './env.js';
import { anthropic, DETERMINISTIC, llmCostUsd, MODEL, SEARCH_USD_PER_CALL } from './llm.js';
import { webSearch } from './search.js';
import { isTimeSensitive } from './searchCache.js';
import { recallMemories, saveMemory } from './memory.js';
import { EMBED_USD_PER_MTOK } from './embeddings.js';
import { bestPassage, fetchPage, focusedExcerpt, type FetchedPage } from './fetchPage.js';
import type { SseStream } from './sse.js';
import { recordRun } from './runlog.js';
import { searchDocuments, type DocHit } from './retrieve.js';

/**
 * The QUICK gear. Two phases:
 *
 *   1. Gather — a tool loop. Claude picks web_search / fetch_page calls; this code runs them
 *      (in parallel when Claude asks for several at once), emits a `trace` per call, and
 *      enforces the caps. Claude never sees a tool this gear is not allowed to use.
 *   2. Answer — `sources` goes out first, then ONE streaming call with no tools offered,
 *      forwarded token by token. With no tools to call, the model can only write.
 *
 * Why stopped is set explicitly at every exit (`terminated`): no SDK reports it for you.
 */

const QUICK_TOOLS: Anthropic.Tool[] = [
  {
    name: 'web_search',
    description:
      'Search the web. Returns up to 5 results with title, url and a short snippet. Snippets are only for deciding what to read: they can never be cited.',
    input_schema: {
      type: 'object',
      properties: { query: { type: 'string', description: 'A focused search query.' } },
      required: ['query'],
      additionalProperties: false
    }
  },
  {
    name: 'fetch_page',
    description:
      'Download a web page and read its main text. Only pages read with this tool can be cited in the answer.',
    input_schema: {
      type: 'object',
      properties: { url: { type: 'string', description: 'A url from a web_search result.' } },
      required: ['url'],
      additionalProperties: false
    }
  },
  {
    name: 'save_memory',
    description:
      'Save a durable fact or preference about the user to long-term memory, so it applies in every future thread. Use only when the user asks you to remember something, or states a lasting preference about how they want answers. Never save facts from web pages.',
    input_schema: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'One concise statement about the user, e.g. "Prefers answers in British English, under 100 words."'
        }
      },
      required: ['text'],
      additionalProperties: false
    }
  }
];

/** Only offered when a Space is selected: there is nothing to search otherwise. */
const SEARCH_DOCUMENTS_TOOL: Anthropic.Tool = {
  name: 'search_documents',
  description:
    "Search the user's own uploaded documents in the selected Space (hybrid keyword + semantic search). Returns passages with their file name and page or section. These passages can be cited.",
  input_schema: {
    type: 'object',
    properties: { query: { type: 'string', description: 'What to look for in the documents.' } },
    required: ['query'],
    additionalProperties: false
  }
};

export const locatorLabel = (l: Locator) =>
  l.page ? `p. ${l.page}` : l.heading ? `§ ${l.heading}` : l.line ? `line ${l.line}` : '';

/**
 * Document passages as the research step reads them: whole, not trimmed. A trimmed passage
 * hides its own second half, and the model then "can't find" what the document does say.
 * Chunks are ~1 000 characters, so six of them cost ~1 500 tokens.
 */
const docDigest = (hits: DocHit[]) => hits.map((h) => `- ${h.title}, ${locatorLabel(h.locator)}:\n${h.text}`).join('\n\n');

export type Remembered = { id: string; text: string };

/** The user's saved memories, as prompt text. Empty when there are none. */
export const memoryBlock = (memories: Remembered[]) =>
  memories.length
    ? `\n\nWhat you know about this user from saved memories (follow any preferences they express):\n${memories.map((m) => `- ${m.text}`).join('\n')}`
    : '';

/** Time kept back from the gather phase so a capped run still has room to write its answer. */
const ANSWER_RESERVE_SEC = 20;

/**
 * The quick gear's economy. A quick answer is one search, two or three pages and a short
 * answer: Scenario E in SPEC.md ("one search, one fetch, one sentence, $0.004"). Anything
 * that needs more research than this is what the deep gear is for.
 */
/** Web searches per quick answer, enforced: the tool is withdrawn once they are spent. */
const QUICK_MAX_SEARCHES = 2;
/** Pages that are enough: once a turn's fetches bring the total here, the answer starts. */
const QUICK_ENOUGH_PAGES = 2;
/** Characters of each page the answer step reads (the best-matching passages, not the whole page). */
const ANSWER_CHARS_PER_PAGE = 2000;
/** A hard ceiling on the quick answer's length. */
const QUICK_ANSWER_MAX_TOKENS = 700;

/** "Remember that…", "from now on…": a message about the user, which needs no web search of its own. */
const isMemoryRequest = (q: string) =>
  /\b(remember|from now on|for all future|keep in mind|my preference|i prefer|always (answer|reply|respond|use))\b/i.test(q);

const gatherPrompt = (maxToolCalls: number, memories: Remembered[]) =>
  `You are the research step of a web search engine. Today is ${new Date().toISOString().slice(0, 10)}.
Find and read the pages needed to answer the user's question. Always search and read, even when you
think you already know the answer: the answer may only use pages fetched here.
- Passages from the user's own documents (search_documents) count as read sources too.
- If the user asks you to remember something about them, or states a lasting preference, call save_memory
  once with a concise statement of it (unless it is already in the saved memories below). If that is
  all the message asks for, no search is needed: reply DONE after saving.
- Results of a first web search may already be given to you. Pick the 2-3 most relevant results and
  fetch_page them ALL IN ONE TURN (parallel calls). Once two or more pages are read, the answer is written
  automatically, so choose well the first time.
- Search again only if none of the results fit the question. A quick answer allows ${QUICK_MAX_SEARCHES} web searches in total.
- Prefer primary and substantive sources (official sites, docs, reputable publications) over social media,
  forums, and directory/profile pages such as LinkedIn.
- Before your tool calls, write one short sentence saying why you're making them.
- Do not write the answer yourself; a later step writes it from the pages you fetched.
- When you have read enough to answer well, reply with just: DONE
You have a budget of ${maxToolCalls} tool calls in total.${memoryBlock(memories)}`;

const QUICK_LENGTH = `
- This is a quick answer: keep it under about 120 words unless the question genuinely needs more (a
  comparison, steps, or a list). The user can switch to Deep for a fuller treatment.`;

export const ANSWER_PROMPT = `You write the answer for a search engine, using ONLY the numbered sources provided: web pages,
and passages from the user's own documents.
- Cite every factual claim with its source number in square brackets at the end of the sentence, before the
  period, like "Tavily is a search API for AI agents [1]." Use [2][3] for several sources.
- Only cite numbers that appear in the sources. Never invent a source, url, or number.
- If the sources don't answer the question, say so plainly instead of guessing.
- Be concise: lead with the direct answer, then the supporting detail. Short paragraphs or a short list.
- No preamble, and no list of sources at the end (the interface shows them).`;

export const CAPPED_NOTE = `
Research was cut off by its tool-call or time budget before it finished. Answer from what is here, and say briefly what you could not verify.`;

export interface HistoryTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface AskInput {
  query: string;
  requestId: string;
  userId: string;
  threadId: string;
  /** Earlier turns of this thread, oldest first, citations already stripped. */
  history: HistoryTurn[];
  /** Where to look: the web, the Space's documents, or the router decides. */
  mode: AskMode;
  /** The Space to search, already checked to belong to this user. */
  spaceId?: string;
}

export interface AskSummary {
  requestId: string;
  answerId: string;
  toolCalls: number;
  terminated: Terminated;
  tokens: { in: number; out: number };
  costUsd: number;
  searchCached: boolean;
  ttftMs: number | null;
  latencyMs: number;
  danglingCitations: number[];
  /** Web searches attempted (cached or not): the denominator of the cache hit rate. */
  webSearches: number;
  answerText: string;
  sources: Source[];
  /** The `done` event as sent. Absent when the run ended in an error. */
  done?: DoneEvent;
  /** Deep search only: the plan it ran, kept so the answer stays explainable later. */
  plan?: SubQuestion[];
}

/**
 * Does this question lean on the conversation? Short, with a word that points back
 * ("it", "that", "what about…"). Only then does the research step see the history:
 * a self-contained question researched without it issues the same searches every
 * time it is asked, which is what lets a repeat hit the search cache.
 */
export function isFollowUp(query: string): boolean {
  const q = query.trim().toLowerCase();
  if (/^(and|but|so|also|then|what about|how about|why not|same)\b/.test(q)) return true;
  const words = q.split(/[^a-z0-9']+/).filter(Boolean);
  if (words.length > 12) return false;
  const at = words.findIndex((w) => POINTERS.has(w));
  if (at < 0) return false;
  // "How does SerpApi price its requests?": the pointer has a subject inside the question.
  // "How much does it cost?": nothing before it to point at, so it points at the thread.
  return words.slice(0, at).filter((w) => !QUESTION_WORDS.has(w)).length < 2;
}

const POINTERS = new Set(
  "it its it's they them their that this those these he she him her his there former latter more else again above earlier previous".split(' ')
);
const QUESTION_WORDS = new Set(
  'what how why when where which who whom whose is are was were be does do did can could should would will much many the a an and or of to in on for with about'.split(' ')
);

export type ToolCallLog = RunLog['toolCalls'][number];

/** Thrown when an upstream provider failed and no honest answer is possible: becomes a 502. */
export class UpstreamError extends Error {}

export async function runQuickAsk(input: AskInput, sse: SseStream, clientGone: AbortSignal): Promise<AskSummary> {
  const t0 = Date.now();
  const answerId = newId('ans');
  const maxToolCalls = env.maxToolCalls;
  const wallClockMs = env.maxWallClockSec * 1000;
  const gatherSignal = AbortSignal.any([
    clientGone,
    AbortSignal.timeout(Math.max(5, env.maxWallClockSec - ANSWER_RESERVE_SEC) * 1000)
  ]);

  const tokens = { in: 0, out: 0 };
  const toolCalls: ToolCallLog[] = [];
  const pages: FetchedPage[] = [];
  const docHits: DocHit[] = [];
  const addDocHits = (hits: DocHit[]) => {
    for (const h of hits) if (!docHits.some((d) => d.chunkId === h.chunkId)) docHits.push(h);
  };
  // A question about something current must not be answered from a 6-hour-old search.
  const wantsFresh = isTimeSensitive(input.query);
  const searches = { attempted: 0, failed: 0, paid: 0, allCached: true, lastError: '' };
  let embedTokens = 0;
  let recalled: Remembered[] = [];
  const saved: string[] = [];
  let terminated: Terminated = 'done';
  let ttftMs: number | null = null;
  let answerText = '';
  let sources: Source[] = [];

  const addUsage = (u: Anthropic.Usage) => {
    tokens.in += u.input_tokens + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0);
    tokens.out += u.output_tokens;
  };

  /** Run one tool call. Never throws: a failure is a result the model and the trace both see. */
  const runTool = async (use: Anthropic.ToolUseBlock) => {
    const started = Date.now();
    const args = (use.input ?? {}) as Record<string, unknown>;
    try {
      if (use.name === 'web_search') {
        const query = String(args.query ?? '').trim();
        if (!query) throw new Error('web_search needs a non-empty query');
        // Checked and counted before the first await, so parallel calls in one turn cannot all slip past it.
        if (searches.attempted >= QUICK_MAX_SEARCHES) {
          throw new Error(`a quick answer allows ${QUICK_MAX_SEARCHES} web searches; use the results already found`);
        }
        searches.attempted++;
        try {
          const out = await webSearch(query, gatherSignal, wantsFresh);
          if (!out.cached) {
            searches.paid++;
            searches.allCached = false;
          }
          const lines = out.results.map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.content.slice(0, 300)}`);
          return {
            ok: true as const,
            ms: Date.now() - started,
            note: `${out.results.length} results${out.cached ? ' (cached)' : out.bypassed ? ' (cache skipped: time-sensitive)' : ''}`,
            content: lines.length ? lines.join('\n') : 'No results.'
          };
        } catch (err) {
          searches.failed++;
          searches.allCached = false;
          searches.lastError = errorMessage(err);
          throw err;
        }
      }
      if (use.name === 'fetch_page') {
        // Numbered later, in the order the calls were made, not the order downloads finished.
        const page = await fetchPage(String(args.url ?? ''), gatherSignal);
        return { ok: true as const, ms: Date.now() - started, note: '', content: '', page };
      }
      if (use.name === 'save_memory') {
        const m = await saveMemory(input.userId, String(args.text ?? ''), input.threadId, gatherSignal);
        embedTokens += m.embedTokens;
        saved.push(m.text);
        return {
          ok: true as const,
          ms: Date.now() - started,
          note: m.duplicate ? `already remembered (${m.id})` : `saved as ${m.id}`,
          content: `Saved to long-term memory: "${m.text}"`
        };
      }
      if (use.name === 'search_documents') {
        if (!input.spaceId) throw new Error('no Space is selected');
        const r = await searchDocuments(input.spaceId, String(args.query ?? input.query), gatherSignal);
        embedTokens += r.embedTokens;
        addDocHits(r.hits);
        return {
          ok: true as const,
          ms: Date.now() - started,
          note: `${r.hits.length} passages (best match ${r.bestVectorScore.toFixed(2)})`,
          content: r.hits.length ? docDigest(r.hits) : 'No matching passages in the Space.'
        };
      }
      throw new Error(`tool ${use.name} is not available in a quick search`);
    } catch (err) {
      return { ok: false as const, ms: Date.now() - started, note: '', content: '', error: errorMessage(err) };
    }
  };

  try {
    // ---------------------------------------------------------------- phase 1: gather
    const messages: Anthropic.MessageParam[] = [
      ...(isFollowUp(input.query) ? input.history : []),
      { role: 'user', content: input.query }
    ];
    let step = 0;
    let nudged = false;

    // Memory is checked before every answer by the harness, not left to the model: a
    // preference has to apply to questions that have nothing to do with it.
    {
      const started = Date.now();
      let error: string | undefined;
      try {
        const r = await recallMemories(input.userId, input.query, gatherSignal);
        recalled = r.memories;
        embedTokens += r.embedTokens;
      } catch (err) {
        // Visible, not fatal: the answer goes ahead without memories and the trace says why.
        error = errorMessage(err);
      }
      const ms = Date.now() - started;
      step++;
      toolCalls.push(error ? { name: 'recall_memory', ok: false, error, ms } : { name: 'recall_memory', ok: true, ms });
      sse.send('trace', {
        step,
        tool: 'recall_memory',
        input: { query: input.query },
        ok: !error,
        ms,
        reason: `memories are checked before every answer → ${recalled.length ? `${recalled.length} recalled` : 'none saved'}`,
        error
      });
    }

    // With a Space selected, the harness searches it before the model does anything. In
    // docs mode that IS the retrieval; in auto mode it is the router's evidence.
    if (input.spaceId && input.mode !== 'web') {
      const started = Date.now();
      let r: Awaited<ReturnType<typeof searchDocuments>> | null = null;
      let error: string | undefined;
      try {
        r = await searchDocuments(input.spaceId, input.query, gatherSignal);
        embedTokens += r.embedTokens;
      } catch (err) {
        error = errorMessage(err);
        // In docs mode the documents are the only source: their failure fails the answer.
        if (input.mode === 'docs') throw new UpstreamError(`document search failed: ${error}`);
      }
      const ms = Date.now() - started;
      const best = r?.bestVectorScore ?? 0;
      const relevant = !!r && (input.mode === 'docs' || best >= env.autoDocMinScore);
      if (r && relevant) addDocHits(r.hits);
      step++;
      toolCalls.push(error ? { name: 'search_documents', ok: false, error, ms } : { name: 'search_documents', ok: true, ms });
      sse.send('trace', {
        step,
        tool: 'search_documents',
        input: { query: input.query, spaceId: input.spaceId },
        ok: !error,
        ms,
        error,
        reason:
          input.mode === 'docs'
            ? `documents mode → ${r?.hits.length ?? 0} passages from the Space`
            : error
              ? 'auto: the Space could not be searched → using the web'
              : relevant
                ? `auto: the Space matches this question (best ${best.toFixed(2)} ≥ ${env.autoDocMinScore}) → answering from its documents, web available if they fall short`
                : `auto: nothing in the Space matches (best ${best.toFixed(2)} < ${env.autoDocMinScore}) → searching the web`
      });
      // Tell the research step what the documents already cover, so it can stop there.
      if (relevant && input.mode === 'auto' && docHits.length) {
        const last = messages.at(-1)!;
        last.content = `${input.query}\n\nAlready retrieved from the user's own documents:\n${docDigest(docHits)}\n\nIf these passages answer the question, reply DONE without searching the web. Search the web only for what they do not cover.`;
      }
    }

    // The first web search is the question itself, run by the harness. Claude then only has
    // to choose pages, in one turn, instead of spending a turn deciding to search. The query
    // is exactly what the user typed, so a repeated question hits the search cache every
    // time. Skipped where the raw question is the wrong query: a follow-up that leans on the
    // thread ("what about its limits?"), a message that is only about remembering something,
    // and a question the Space's documents already cover.
    const seedWeb =
      input.mode !== 'docs' && docHits.length === 0 && !isFollowUp(input.query) && !isMemoryRequest(input.query);
    if (seedWeb) {
      const seed = await runTool({ type: 'tool_use', id: 'seed_search', name: 'web_search', input: { query: input.query } } as Anthropic.ToolUseBlock);
      step++;
      toolCalls.push(seed.ok ? { name: 'web_search', ok: true, ms: seed.ms } : { name: 'web_search', ok: false, error: seed.error, ms: seed.ms });
      sse.send('trace', {
        step,
        tool: 'web_search',
        input: { query: input.query },
        ok: seed.ok,
        ms: seed.ms,
        reason: `a quick search starts with the question as asked → ${seed.ok ? seed.note : 'failed'}`,
        error: seed.ok ? undefined : seed.error
      });
      const last = messages.at(-1)!;
      last.content = seed.ok
        ? `${input.query}\n\nWeb search results for this question (already run):\n${seed.content}\n\nFetch the 2-3 best of these in one turn. Search again only if none of them fit.`
        : `${input.query}\n\n(A first web search for this question failed: ${seed.error}. Try a search yourself.)`;
    }

    // Documents mode does not research the web: the Space search above is the retrieval.
    gather: while (input.mode !== 'docs') {
      let reply: Anthropic.Message;
      try {
        reply = await anthropic.messages.create(
          {
            model: MODEL,
            max_tokens: 1024,
            system: gatherPrompt(maxToolCalls - toolCalls.length, recalled),
            // Searches spent: the tool is withdrawn, so the limit is a fact, not a request.
            tools: [
              ...QUICK_TOOLS.filter((t) => t.name !== 'web_search' || searches.attempted < QUICK_MAX_SEARCHES),
              ...(input.spaceId ? [SEARCH_DOCUMENTS_TOOL] : [])
            ],
            messages,
            ...DETERMINISTIC
          },
          { signal: gatherSignal }
        );
      } catch (err) {
        if (isOurDeadline(gatherSignal, clientGone)) {
          terminated = 'cap';
          break gather;
        }
        throw err;
      }
      addUsage(reply.usage);

      const uses = reply.content.filter((b): b is Anthropic.ToolUseBlock => b.type === 'tool_use');
      if (uses.length === 0) {
        // The model says it has enough. On an "easy" question it may say so before reading
        // anything, which would mean answering from memory: send it back once.
        // A message that only asked to be remembered is the exception: it needs no sources.
        if (pages.length + docHits.length === 0 && saved.length === 0 && !nudged && toolCalls.length < maxToolCalls) {
          nudged = true;
          messages.push({ role: 'assistant', content: reply.content });
          messages.push({
            role: 'user',
            content:
              'You have not read any pages yet, and the answer may only use pages fetched in this request, however well you think you know it. Search and fetch pages before saying DONE.'
          });
          continue gather;
        }
        break gather;
      }

      const remaining = maxToolCalls - toolCalls.length;
      if (remaining <= 0) {
        // The (maxToolCalls+1)th call: denied, never run.
        terminated = 'cap';
        break gather;
      }
      const allowed = uses.slice(0, remaining);
      const why = reply.content
        .filter((b): b is Anthropic.TextBlock => b.type === 'text')
        .map((b) => b.text.trim())
        .join(' ')
        .slice(0, 300);

      const results = await Promise.all(allowed.map(runTool));
      for (const r of results) {
        if (!r.ok || !('page' in r) || !r.page) continue;
        const page = r.page;
        let n = pages.findIndex((p) => p.url === page.url) + 1;
        if (!n) n = pages.push(page);
        r.note = `read as source [${n}]: ${page.title}`;
        // Only seen if research continues; the answer step gets its own excerpt of the page.
        r.content = `Source [${n}]: ${page.title}\n${page.url}\n\n${focusedExcerpt(page.text, input.query, 1200)}`;
      }

      allowed.forEach((use, i) => {
        const r = results[i]!;
        step++;
        toolCalls.push(r.ok ? { name: use.name as AskTool, ok: true, ms: r.ms } : { name: use.name as AskTool, ok: false, error: r.error, ms: r.ms });
        sse.send('trace', {
          step,
          tool: use.name,
          input: use.input as Record<string, unknown>,
          ok: r.ok,
          ms: r.ms,
          reason: [why, r.note].filter(Boolean).join(' → ') || undefined,
          error: r.ok ? undefined : r.error
        });
      });

      if (clientGone.aborted) throw new Error('client disconnected');
      if (gatherSignal.aborted || allowed.length < uses.length) {
        terminated = 'cap';
        break gather;
      }
      // Enough pages read: write the answer now rather than paying for a turn that says DONE.
      if (pages.length >= QUICK_ENOUGH_PAGES && allowed.some((u) => u.name === 'fetch_page')) break gather;

      messages.push({ role: 'assistant', content: reply.content });
      messages.push({
        role: 'user',
        content: allowed.map((use, i) => {
          const r = results[i]!;
          return {
            type: 'tool_result' as const,
            tool_use_id: use.id,
            content: r.ok ? r.content : `Error: ${r.error}`,
            is_error: !r.ok
          };
        })
      });
    }

    // ---------------------------------------------------------------- phase 2: answer
    const totalSources = pages.length + docHits.length;
    if (totalSources === 0) {
      // Every search threw: the provider is down. Saying "nothing found" would be a lie.
      if (searches.attempted > 0 && searches.failed === searches.attempted) {
        throw new UpstreamError(`search provider failed: ${searches.lastError}`);
      }
      sources = [];
      sse.send('sources', sources);
      answerText = saved.length
        ? `Got it. I'll remember: ${saved.map((s) => `"${s}"`).join(' ')} It applies to all your threads; you can see or delete it under Memory.`
        : input.mode === 'docs'
          ? "I couldn't find anything relevant to this in the Space's documents, so I can't answer it from them. Try rephrasing, or switch to web search."
          : terminated === 'cap'
          ? "I ran out of my research budget before I could read any pages, so I can't give a sourced answer. Try a narrower question."
          : searches.attempted === 0
            ? "I didn't run any searches for this question, so I have nothing to cite. Try rephrasing it."
            : "I searched but couldn't read any of the pages I found (they were blocked or unreadable), so I can't give a sourced answer. Try rephrasing the question.";
      ttftMs = Date.now() - t0;
      sse.send('token', { text: answerText });
    } else {
      // One numbering for everything: web pages first (their numbers match the trace),
      // then document passages in retrieval rank.
      sources = [
        ...pages.map((p, i) => ({
          n: i + 1,
          kind: 'web' as const,
          title: p.title,
          url: p.url,
          snippet: bestPassage(p.text, input.query)
        })),
        ...docHits.map((h, i) => ({
          n: pages.length + i + 1,
          kind: 'doc' as const,
          title: h.title,
          docId: h.docId as Source['docId'],
          locator: h.locator,
          // The whole chunk: it is exactly the passage the answer was given, so the citation
          // quotes what was retrieved, not a summary of it.
          snippet: h.text
        }))
      ];
      sse.send('sources', sources); // before the first token, always

      // Each page as the best-matching ~2 000 characters, not the whole page. The citation's
      // snippet is always among them, so what a [n] quotes is text the answer actually read.
      const excerptOf = (p: FetchedPage, snippet: string) => {
        const excerpt = focusedExcerpt(p.text, input.query, ANSWER_CHARS_PER_PAGE - snippet.length);
        return excerpt.includes(snippet) ? excerpt : `${excerpt} … ${snippet}`;
      };
      const sourceBlock = [
        ...pages.map((p, i) => `[${i + 1}] ${p.title} — ${p.url}\n${excerptOf(p, sources[i]!.snippet)}`),
        ...docHits.map((h, i) => `[${pages.length + i + 1}] ${h.title}, ${locatorLabel(h.locator)} (the user's document)\n${h.text}`)
      ].join('\n\n');
      const answerSignal = AbortSignal.any([
        clientGone,
        AbortSignal.timeout(Math.max(10_000, wallClockMs - (Date.now() - t0)))
      ]);

      try {
        const stream = anthropic.messages.stream(
          {
            model: MODEL,
            max_tokens: QUICK_ANSWER_MAX_TOKENS,
            system:
              ANSWER_PROMPT +
              QUICK_LENGTH +
              (terminated === 'cap' ? CAPPED_NOTE : '') +
              memoryBlock([...recalled, ...saved.map((text) => ({ id: 'new', text }))]),
            messages: [
              // The answer always sees the conversation, so a follow-up reads as one. Earlier
              // answers had their [n]s stripped: only this request's sources may be cited.
              ...input.history,
              {
                role: 'user',
                content: `Question: ${input.query}\n\n<sources>\n${sourceBlock}\n</sources>\n\nOnly cite numbers 1-${totalSources}.`
              }
            ]
          },
          { signal: answerSignal }
        );
        for await (const ev of stream) {
          if (ev.type === 'content_block_delta' && ev.delta.type === 'text_delta') {
            if (ttftMs === null) ttftMs = Date.now() - t0;
            answerText += ev.delta.text;
            sse.send('token', { text: ev.delta.text });
          }
        }
        addUsage((await stream.finalMessage()).usage);
      } catch (err) {
        if (!isOurDeadline(answerSignal, clientGone)) throw err;
        terminated = 'cap'; // the answer ran out of wall clock mid-stream: an honest partial
      }
    }

    const summary = finish();
    summary.done = {
      answerId,
      latencyMs: summary.latencyMs,
      ttftMs: ttftMs ?? summary.latencyMs,
      model: MODEL,
      tokens: { ...tokens },
      costUsd: summary.costUsd,
      searchCached: summary.searchCached,
      terminated,
      depth: 'quick',
      subQuestions: 0
    };
    sse.send('done', summary.done);
    await writeRunLog(input, summary, toolCalls, false);
    return summary;
  } catch (err) {
    terminated = 'error';
    const summary = finish();
    const gone = clientGone.aborted;
    await writeRunLog(input, summary, toolCalls, gone);
    if (gone) return summary;
    throw err instanceof UpstreamError ? err : new UpstreamError(errorMessage(err));
  }

  function finish(): AskSummary {
    const latencyMs = Date.now() - t0;
    const costUsd = round6(
      llmCostUsd(MODEL, tokens.in, tokens.out) +
        searches.paid * SEARCH_USD_PER_CALL +
        (embedTokens / 1e6) * EMBED_USD_PER_MTOK
    );
    return {
      requestId: input.requestId,
      answerId,
      toolCalls: toolCalls.length,
      terminated,
      tokens: { ...tokens },
      costUsd,
      // True only when every search in the request was a cache hit (and there was at least one).
      searchCached: searches.attempted > 0 && searches.allCached,
      webSearches: searches.attempted,
      ttftMs,
      latencyMs,
      danglingCitations: unresolvedCitations(answerText, sources),
      answerText,
      sources
    };
  }
}

/** True when a signal fired because OUR wall-clock budget ran out, not because the user left. */
export function isOurDeadline(signal: AbortSignal, clientGone: AbortSignal) {
  return signal.aborted && !clientGone.aborted;
}

export function writeRunLog(
  input: AskInput,
  s: AskSummary,
  toolCalls: ToolCallLog[],
  aborted: boolean,
  depth: 'quick' | 'deep' = 'quick'
) {
  return recordRun({
    requestId: input.requestId,
    userId: input.userId,
    threadId: input.threadId,
    answerId: s.answerId,
    query: input.query,
    depth,
    aborted,
    stats: { ttftMs: s.ttftMs, searchCached: s.searchCached, webSearches: s.webSearches },
    run: {
      tokens: s.tokens.in + s.tokens.out,
      wallClockSec: Math.round(s.latencyMs / 100) / 10,
      costUsd: s.costUsd,
      terminated: s.terminated,
      toolCalls
    }
  });
}

export const errorMessage = (err: unknown) => (err instanceof Error ? err.message : String(err)) || 'unknown error';
export const round6 = (n: number) => Math.round(n * 1e6) / 1e6;
