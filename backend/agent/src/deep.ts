import Anthropic from '@anthropic-ai/sdk';
import { newId, PlanEvent, unresolvedCitations, type AskTool, type Source, type SubQuestion, type Terminated } from '@lumina/contract';
import { env } from './env.js';
import { anthropic, DETERMINISTIC, llmCostUsd, MODEL, SEARCH_USD_PER_CALL } from './llm.js';
import { EMBED_USD_PER_MTOK } from './embeddings.js';
import { webSearch } from './search.js';
import { isTimeSensitive } from './searchCache.js';
import { bestPassage, fetchPage, type FetchedPage } from './fetchPage.js';
import { recallMemories } from './memory.js';
import { searchDocuments, type DocHit } from './retrieve.js';
import {
  ANSWER_PROMPT,
  CAPPED_NOTE,
  errorMessage,
  isFollowUp,
  isOurDeadline,
  locatorLabel,
  memoryBlock,
  round6,
  UpstreamError,
  writeRunLog,
  type AskInput,
  type AskSummary,
  type Remembered,
  type ToolCallLog
} from './loop.js';
import type { SseStream } from './sse.js';

/**
 * The DEEP gear, Perplexity's Pro Search:
 *
 *   recall_memory → plan_research (stream `plan` BEFORE any retrieval)
 *     → fan-out: every sub-question researched in parallel (search → fetch its best pages)
 *     → merge into ONE contiguous citation numbering, every step and source tagged with
 *       the sub-question it served → one structured, streamed answer.
 *
 * Only reachable with depth: "deep", which the user opts into and which has already passed
 * the daily spend gate (deepCap.ts). The quick gear never sees plan_research.
 *
 * The fan-out is deterministic (search the sub-question, read its top results), not a model
 * loop per sub-question: that keeps the sub-questions parallel, the cost predictable, and
 * the 24-call budget allocatable up front.
 */

/** Time kept back from research for writing the long, structured answer. */
const ANSWER_RESERVE_SEC = 60;
/** Pages read per sub-question, at most. */
const PAGES_PER_SUB = 3;
/** Fetch attempts per sub-question, failures included: enough to replace one blocked page, not to thrash. */
const FETCH_ATTEMPTS_PER_SUB = PAGES_PER_SUB + 1;

const planPrompt = (min: number, max: number) =>
  `You plan the research for a search engine's deep mode. Today is ${new Date().toISOString().slice(0, 10)}.
Break the user's question into ${min}-${max} sub-questions that together answer it fully.
- Each must be a question a knowledgeable person would actually ask: specific, self-contained (no "it" or
  "they" pointing back), and answerable with a web search.
- They must not overlap. Each should send the research to different pages. Cover every part of the
  question; when it compares options, give each option or axis its own sub-question instead of
  restating the whole comparison.
- Give each a reason of at most 12 words: what it contributes to the final answer.
- Give each a "search": a 3-8 word keyword query a search engine handles well (no question words,
  no filler), e.g. "SSE buffering CDN proxy issues".
Reply with ONLY a JSON object, no prose and no code fence:
{"reason": "<at most 15 words on how you split it>", "subQuestions": [{"question": "...", "reason": "...", "search": "..."}]}`;

/** Results the provider scores below this are not worth a fetch (news and press releases that merely share a word). */
const MIN_RESULT_SCORE = 0.3;

const DEEP_STRUCTURE = `
This is a deep-research answer. Structure it as:
1. A direct answer to the whole question in 2-4 sentences, with citations.
2. One section per sub-question, headed "### <the sub-question>", answering it from the sources found for it.
3. A final "### What's still unclear" section naming the specific gaps or disagreements the sources left open.
   If there are none, say so in one line. Never pad it.
Be thorough, not padded: do not repeat a point across sections.`;

type Found = { sub: number } & ({ kind: 'web'; page: FetchedPage } | { kind: 'doc'; hit: DocHit });

export async function runDeepAsk(input: AskInput, sse: SseStream, clientGone: AbortSignal): Promise<AskSummary> {
  const t0 = Date.now();
  const answerId = newId('ans');
  const maxCalls = env.maxToolCallsDeep;
  const wallClockMs = env.maxWallClockSecDeep * 1000;
  const researchSignal = AbortSignal.any([
    clientGone,
    AbortSignal.timeout(Math.max(30, env.maxWallClockSecDeep - ANSWER_RESERVE_SEC) * 1000)
  ]);
  const wantsFresh = isTimeSensitive(input.query);

  const tokens = { in: 0, out: 0 };
  const toolCalls: ToolCallLog[] = [];
  /** The sub-question each tool call served (0 = the whole question), parallel to toolCalls. */
  const callSub: number[] = [];
  const found: Found[] = [];
  const searches = { attempted: 0, failed: 0, paid: 0, allCached: true, lastError: '' };
  let embedTokens = 0;
  let recalled: Remembered[] = [];
  let terminated: Terminated = 'done';
  let ttftMs: number | null = null;
  let answerText = '';
  let sources: Source[] = [];
  let plan: SubQuestion[] = [];
  /** The planner's keyword query for each sub-question: what the web search actually sends. */
  const searchFor = new Map<number, string>();
  let step = 0;
  /** Tool calls started (not just finished): parallel workers check this BEFORE starting one. */
  let spent = 0;

  const addUsage = (u: Anthropic.Usage) => {
    tokens.in += u.input_tokens + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0);
    tokens.out += u.output_tokens;
  };

  const emit = (
    tool: AskTool,
    toolInput: Record<string, unknown>,
    r: { ok: boolean; ms: number; error?: string; reason?: string },
    subQuestion?: number
  ) => {
    step++;
    toolCalls.push(r.ok ? { name: tool, ok: true, ms: r.ms } : { name: tool, ok: false, error: r.error || 'unknown error', ms: r.ms });
    callSub.push(subQuestion ?? 0);
    sse.send('trace', {
      step,
      tool,
      input: toolInput,
      ok: r.ok,
      ms: r.ms,
      reason: r.reason,
      error: r.ok ? undefined : r.error,
      subQuestion
    });
  };

  try {
    // ---------------------------------------------------------------- memory
    {
      const started = Date.now();
      spent++;
      let error: string | undefined;
      try {
        const r = await recallMemories(input.userId, input.query, researchSignal);
        recalled = r.memories;
        embedTokens += r.embedTokens;
      } catch (err) {
        error = errorMessage(err);
      }
      emit('recall_memory', { query: input.query }, {
        ok: !error,
        ms: Date.now() - started,
        error,
        reason: `memories are checked before every answer → ${recalled.length ? `${recalled.length} recalled` : 'none saved'}`
      });
    }

    // ---------------------------------------------------------------- plan (before ANY retrieval)
    {
      const started = Date.now();
      spent++;
      const made = await makePlan();
      plan = made.subQuestions;
      sse.send('plan', { subQuestions: plan, reason: made.reason }); // the deep gear's first paint
      emit('plan_research', { query: input.query }, {
        ok: true,
        ms: Date.now() - started,
        reason: `${plan.length} sub-questions${made.reason ? `: ${made.reason}` : ''}`
      });
    }

    // ---------------------------------------------------------------- fan-out
    // The remaining budget, split evenly up front, so parallel sub-questions cannot overrun it.
    const allowance = Math.floor((maxCalls - spent) / plan.length);
    const claimedUrls = new Set<string>();
    await Promise.all(plan.map((sq) => researchSub(sq, allowance, claimedUrls)));
    if (clientGone.aborted) throw new Error('client disconnected');
    if (researchSignal.aborted) terminated = 'cap'; // the research hit its wall-clock budget

    // ---------------------------------------------------------------- merge
    // One numbering, grouped by sub-question in plan order, discovery order within each.
    const merged = [...found].sort((a, b) => a.sub - b.sub);
    if (merged.length === 0) {
      if (searches.attempted > 0 && searches.failed === searches.attempted) {
        throw new UpstreamError(`search provider failed: ${searches.lastError}`);
      }
    }
    sources = merged.map((f, k) =>
      f.kind === 'web'
        ? {
            n: k + 1,
            kind: 'web' as const,
            title: f.page.title,
            url: f.page.url,
            snippet: bestPassage(f.page.text, plan[f.sub - 1]!.question),
            subQuestion: f.sub
          }
        : {
            n: k + 1,
            kind: 'doc' as const,
            title: f.hit.title,
            docId: f.hit.docId as Source['docId'],
            locator: f.hit.locator,
            snippet: f.hit.text,
            subQuestion: f.sub
          }
    );
    sse.send('sources', sources); // before the first token

    // ---------------------------------------------------------------- answer
    if (merged.length === 0) {
      answerText =
        terminated === 'cap'
          ? "I ran out of my research budget before I could read any sources, so I can't give a sourced answer."
          : "I researched every part of the plan but couldn't read any usable sources (pages were blocked or unreadable), so I can't give a sourced answer. Try rephrasing the question.";
      ttftMs = Date.now() - t0;
      sse.send('token', { text: answerText });
    } else {
      const planList = plan.map((s) => `${s.i}. ${s.question}`).join('\n');
      const sourceBlock = merged
        .map((f, k) =>
          f.kind === 'web'
            ? `[${k + 1}] (sub-question ${f.sub}) ${f.page.title} — ${f.page.url}\n${f.page.text.slice(0, 4000)}`
            : `[${k + 1}] (sub-question ${f.sub}) ${f.hit.title}, ${locatorLabel(f.hit.locator)} (the user's document)\n${f.hit.text}`
        )
        .join('\n\n');
      const answerSignal = AbortSignal.any([
        clientGone,
        AbortSignal.timeout(Math.max(20_000, wallClockMs - (Date.now() - t0)))
      ]);
      try {
        const stream = anthropic.messages.stream(
          {
            model: MODEL,
            max_tokens: 4096,
            system: ANSWER_PROMPT + DEEP_STRUCTURE + (terminated === 'cap' ? CAPPED_NOTE : '') + memoryBlock(recalled),
            messages: [
              ...input.history,
              {
                role: 'user',
                content: `Question: ${input.query}\n\nThe research plan:\n${planList}\n\n<sources>\n${sourceBlock}\n</sources>\n\nOnly cite numbers 1-${merged.length}.`
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
        terminated = 'cap';
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
      depth: 'deep',
      subQuestions: plan.length
    };
    sse.send('done', summary.done);
    await writeRunLog(input, summary, trajectory(), false, 'deep');
    return summary;
  } catch (err) {
    terminated = 'error';
    const summary = finish();
    const gone = clientGone.aborted;
    await writeRunLog(input, summary, trajectory(), gone, 'deep');
    if (gone) return summary;
    throw err instanceof UpstreamError ? err : new UpstreamError(errorMessage(err));
  }

  // ------------------------------------------------------------------ helpers

  /** Ask the planner for 3-6 sub-questions. One retry if the plan is malformed or too small. */
  async function makePlan(): Promise<{ reason?: string; subQuestions: SubQuestion[] }> {
    const min = env.deepSubQuestionsMin;
    const max = env.deepSubQuestionsMax;
    const messages: Anthropic.MessageParam[] = [
      ...(isFollowUp(input.query) ? input.history : []),
      { role: 'user', content: input.query }
    ];
    let problem = '';
    for (let attempt = 0; attempt < 2; attempt++) {
      const reply = await anthropic.messages.create(
        { model: MODEL, max_tokens: 1024, system: planPrompt(min, max) + memoryBlock(recalled), messages, ...DETERMINISTIC },
        { signal: researchSignal }
      );
      addUsage(reply.usage);
      const text = reply.content
        .filter((b): b is Anthropic.TextBlock => b.type === 'text')
        .map((b) => b.text)
        .join('');
      try {
        const json = JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] ?? '') as {
          reason?: string;
          subQuestions?: Array<{ question?: string; reason?: string; search?: string }>;
        };
        const raw = (json.subQuestions ?? []).filter((s) => typeof s.question === 'string' && s.question.trim()).slice(0, max);
        const subs = raw.map((s, i) => ({ i: i + 1, question: s.question!.trim(), ...(s.reason ? { reason: s.reason.trim() } : {}) }));
        if (subs.length >= min) {
          // The keyword query is the planner's, but it is not part of the plan event's contract.
          raw.forEach((s, i) => searchFor.set(i + 1, s.search?.trim() || s.question!.trim()));
          const parsed = PlanEvent.parse({ subQuestions: subs, reason: json.reason?.trim() || undefined });
          return { reason: parsed.reason, subQuestions: parsed.subQuestions };
        }
        problem = `That plan had ${subs.length} sub-questions; it needs between ${min} and ${max}.`;
      } catch {
        problem = 'That was not valid JSON in the requested shape.';
      }
      messages.push({ role: 'assistant', content: text || '(empty)' });
      messages.push({ role: 'user', content: `${problem} Reply again with ONLY the JSON object.` });
    }
    throw new UpstreamError(`the planner did not return a usable plan: ${problem}`);
  }

  /** Research one sub-question within its allowance of tool calls. Never throws. */
  async function researchSub(sq: SubQuestion, allowance: number, claimedUrls: Set<string>) {
    let left = allowance;
    const take = () => {
      if (left <= 0 || spent >= maxCalls || researchSignal.aborted) return false;
      left--;
      spent++;
      return true;
    };

    // The user's documents, when a Space is in play.
    if (input.spaceId && input.mode !== 'web' && take()) {
      const started = Date.now();
      try {
        const r = await searchDocuments(input.spaceId, sq.question, researchSignal, 3);
        embedTokens += r.embedTokens;
        const relevant = input.mode === 'docs' || r.bestVectorScore >= env.autoDocMinScore;
        if (relevant) {
          for (const hit of r.hits) {
            if (!found.some((f) => f.kind === 'doc' && f.hit.chunkId === hit.chunkId)) found.push({ sub: sq.i, kind: 'doc', hit });
          }
        }
        emit('search_documents', { query: sq.question, spaceId: input.spaceId }, {
          ok: true,
          ms: Date.now() - started,
          reason: `${r.hits.length} passages (best match ${r.bestVectorScore.toFixed(2)})${relevant ? '' : ' → not relevant, not used'}`
        }, sq.i);
      } catch (err) {
        emit('search_documents', { query: sq.question, spaceId: input.spaceId }, { ok: false, ms: Date.now() - started, error: errorMessage(err) }, sq.i);
      }
    }
    if (input.mode === 'docs' || !take()) return;

    // The web: search with the planner's keyword query, then read the best results.
    const query = searchFor.get(sq.i) ?? sq.question;
    const started = Date.now();
    searches.attempted++;
    let results: { title: string; url: string }[] = [];
    try {
      const out = await webSearch(query, researchSignal, wantsFresh);
      if (!out.cached) {
        searches.paid++;
        searches.allCached = false;
      }
      // A low provider score means the page shares words with the query, not its subject.
      results = out.results.filter((r) => r.score === undefined || r.score >= MIN_RESULT_SCORE);
      const dropped = out.results.length - results.length;
      emit('web_search', { query }, {
        ok: true,
        ms: Date.now() - started,
        reason: `researching sub-question ${sq.i} → ${out.results.length} results${out.cached ? ' (cached)' : ''}${dropped ? `, ${dropped} below relevance ${MIN_RESULT_SCORE} skipped` : ''}`
      }, sq.i);
    } catch (err) {
      searches.failed++;
      searches.allCached = false;
      searches.lastError = errorMessage(err);
      emit('web_search', { query }, { ok: false, ms: Date.now() - started, error: searches.lastError }, sq.i);
      return;
    }

    // Up to PAGES_PER_SUB readers in parallel. A page another sub-question already claimed
    // is skipped (that is what makes deep read MORE, not the same pages twice); a failed
    // fetch moves on to the next result while the allowance lasts.
    let next = 0;
    let attempts = 0;
    const reader = async () => {
      while (next < results.length && attempts < FETCH_ATTEMPTS_PER_SUB) {
        const candidate = results[next++]!;
        if (claimedUrls.has(candidate.url)) continue;
        if (!take()) return;
        attempts++;
        claimedUrls.add(candidate.url);
        const t = Date.now();
        try {
          const page = await fetchPage(candidate.url, researchSignal);
          const duplicate = found.some((f) => f.kind === 'web' && f.page.url === page.url);
          if (!duplicate) found.push({ sub: sq.i, kind: 'web', page });
          emit('fetch_page', { url: candidate.url }, {
            ok: true,
            ms: Date.now() - t,
            reason: duplicate ? 'already read for another sub-question' : `read for sub-question ${sq.i}: ${page.title}`
          }, sq.i);
          if (!duplicate) return;
        } catch (err) {
          emit('fetch_page', { url: candidate.url }, { ok: false, ms: Date.now() - t, error: errorMessage(err) }, sq.i);
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(PAGES_PER_SUB, left) }, reader));
  }

  /**
   * The run log's tool order. The sub-questions ran in parallel, so completion order is an
   * accident of timing; the log groups each branch's calls together (whole-question steps
   * first, then sub-question 1, 2, …), which is the trajectory a reader can follow. The
   * streamed trace keeps the real completion order.
   */
  function trajectory(): ToolCallLog[] {
    return toolCalls
      .map((call, k) => ({ call, sub: callSub[k] ?? 0, k }))
      .sort((a, b) => a.sub - b.sub || a.k - b.k)
      .map((x) => x.call);
  }

  function finish(): AskSummary {
    const latencyMs = Date.now() - t0;
    return {
      requestId: input.requestId,
      answerId,
      toolCalls: toolCalls.length,
      terminated,
      tokens: { ...tokens },
      costUsd: round6(
        llmCostUsd(MODEL, tokens.in, tokens.out) + searches.paid * SEARCH_USD_PER_CALL + (embedTokens / 1e6) * EMBED_USD_PER_MTOK
      ),
      searchCached: searches.attempted > 0 && searches.allCached,
      webSearches: searches.attempted,
      ttftMs,
      latencyMs,
      danglingCitations: unresolvedCitations(answerText, sources),
      answerText,
      sources,
      plan
    };
  }
}
