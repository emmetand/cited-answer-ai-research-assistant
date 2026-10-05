# Architecture

How Cited is put together, what each piece is allowed to do, what happens when things fail,
and the trade-offs behind the design.

## Components

| Component | Runs as | Job |
|---|---|---|
| **UI** | React + Vite, static | Asks, renders the stream, shows citations, plan, trace, memory and Spaces |
| **Gateway** | Express, public | The edge: identity, validation, rate limiting, proxying, stream pass-through |
| **Agent service** | Express, private | The agent loop and tools, quick and deep search, memory, retrieval, the spend gate |
| **Jobs worker** | Separate Node process | Turns uploaded documents into searchable chunks |
| **MongoDB Atlas** | Managed | All state, vectors included |
| **Providers** | External APIs | Claude (answers), OpenAI (embeddings), Tavily or SerpApi (search) |

Not services, but components all the same, because they hold state or make decisions:
the **search cache** (in-process LRU over a TTL'd collection), the **jobs queue** (a MongoDB
collection), the **daily deep-search counter**, and the **run logs**.

## Responsibilities

The useful sentences are the exclusions.

- **Only the agent service holds provider keys.** The gateway never calls a model, a search
  API or an embedding API, and nothing a browser can load contains a key.
- **Only the gateway talks to the browser.** The agent service should not be publicly
  reachable.
- **The agent service enforces the deep-search daily cap, not the gateway.** The cap guards
  spending, so it sits next to the spending; a cap at the edge is a cap you bypass by calling
  the service behind it.
- **The gateway enforces the per-minute rate limit.** That protects the service as a whole
  and needs no provider knowledge.
- **Only the user chooses deep.** Quick is the default, the server never upgrades a request,
  and a quick run is never given the `plan_research` tool, so the model *can't* escalate,
  whatever it would like to do.
- **Only the worker parses, chunks and embeds documents.** The upload route stores the file,
  queues a job and returns `202`.
- **The loop, not the model, owns the limits.** Tool-call and time caps, the quick mode's
  search budget, and when research stops are all decided in code.

## Request lifecycles

### Quick answer
1. **Recall memory** (harness step, every request): a vector search plus the user's newest
   memories, so preferences apply even to unrelated questions.
2. **Documents** (when a Space is selected): hybrid search. In `docs` mode this *is* the
   retrieval; in `auto` mode the best vector score decides whether the documents are relevant.
3. **Seed search:** the question itself is searched by the harness, verbatim, which makes
   repeats cache-friendly and saves a model turn.
4. **One research turn:** the model picks the 2–3 best results and fetches them in parallel.
   Once two pages are read, research stops. It can search once more if nothing fit.
5. **`sources`** stream, then **one streaming model call with no tools** writes the answer from
   the best-matching passages of each page. With nothing to call, it can only write.
6. **`done`** reports latency, time to first token, tokens, cost, whether search was cached,
   and why the run stopped (`done`, `cap` or `error`).

### Deep answer
1. **Spend gate:** an atomic per-user daily counter; over the cap is a `429` with `resetsAt`.
2. Recall memory, then **plan**: one model call returns 3–6 sub-questions, each with a reason
   and a keyword search query. The `plan` event streams before any retrieval.
3. **Fan-out:** sub-questions are researched in parallel. The 24-call budget is split between
   them up front, so parallel branches can't overrun it. Each searches its keyword query,
   skips low-relevance results and pages another branch already claimed, and reads up to
   three pages.
4. **Merge:** one contiguous numbering, every trace step and source tagged with its sub-question.
5. **Answer:** a structured synthesis — direct answer, a section per sub-question, and the gaps
   the sources left open.

### Document ingestion
`upload → 202` (file in GridFS, document `pending`, job queued) → the worker claims the job
atomically → **parse** (page-aware PDF, heading-aware Markdown) → **chunk** (~1 000 characters,
sentence-aware, never across a page) → **embed** → **upsert** → **probe** both search indexes
until they return the new chunks → `indexed`.

## Communication and failure

| Link | How | When the other side is down |
|---|---|---|
| Browser → gateway | HTTP; answers as Server-Sent Events | — |
| Gateway → agent | Same contract, proxied; streams forwarded chunk by chunk | `502` with the reason |
| Agent → worker | Never directly: a row in the `jobs` collection | Uploads still succeed and wait in the queue |
| Agent → providers | HTTPS with timeouts | See below |

- **A provider fails before the stream starts:** HTTP `502`.
- **It fails mid-stream:** an `error` event and `terminated: "error"`. Never a plausible answer
  built around the hole; a silent fallback turns a broken dependency into confident wrong
  output that nothing alerts on.
- **Every search fails:** `502`. Saying "nothing found" when the real cause was an exception
  would be a lie.
- **A page can't be fetched:** a failed trace step with the reason; research continues.
- **A cap is hit:** the answer is written from what was gathered, says what it couldn't
  verify, and reports `terminated: "cap"`.
- **The user closes the tab:** the gateway cancels the upstream request and the agent stops
  spending. The run is logged separately, because it isn't a failure of the agent.
- **The worker dies mid-job:** the row stays `running` with a stale claim; a sweeper returns it
  to `pending`, and the retry reuses stored embeddings.
- **Memory recall fails:** a failed trace step; the answer goes ahead without memories.
- **The search cache is unreachable:** a cache miss. It's a cache, never a source of truth.

## State

| Data | Where | Authoritative? |
|---|---|---|
| Threads, messages | MongoDB | Yes |
| Memories (text + embedding) | MongoDB, vector index filtered by user | Yes. The collection is the whole truth, and deleting the row deletes the memory |
| Spaces, documents, raw uploads | MongoDB, GridFS | Yes |
| Chunks (text, locator, embedding) | MongoDB, vector + text indexes filtered by Space | Derived: rebuildable from the uploads, at a cost in time and embedding spend |
| Jobs | MongoDB | Yes, until done |
| Search results | In-process LRU, then a TTL'd collection (6 h) | No: deleting it only costs money and time |
| Deep-search counter | MongoDB, one document per user per day | Yes |
| Run logs | Local disk and MongoDB | Yes, as evidence |
| Rate-limit window | Gateway memory | No: lost on restart |

**Written but not yet searchable.** After chunks are written, a document stays `embedding`
until a probe query finds them in both search indexes. Only then does it become `indexed`.

## Trade-offs

- **Atlas Vector Search instead of a dedicated vector database.** A citation is one document,
  a Space is a plain filter, and there's one store to back up. The cost: the free tier allows
  exactly three search indexes, and Atlas Search lags writes by seconds (hence the probe).
- **Two services instead of one.** Keys stay off the edge and each side fails independently.
  The cost: two deployables and an extra network hop. Measured, the hop adds no visible
  latency to uploads.
- **Deterministic deep fan-out instead of a model loop per sub-question.** Branches run in
  parallel, cost is predictable and the budget can be split up front. The cost: a branch
  can't adapt mid-research. The planner writes each branch's search query, which recovered
  most of the source quality a per-branch loop would add.
- **A harness-run seed search for quick answers.** It saves a model turn and makes repeat
  questions hit the cache. The cost: the model can't rephrase the first query. Follow-ups and
  memory-only messages skip the seed for exactly that reason.
- **Rate limits in memory.** Simple and fast, but per instance and lost on restart. Fine for
  one gateway; a shared store would be needed for more.
- **No re-ranker.** Reciprocal rank fusion over two retrievers already reaches 39/39 recall@5
  on the gold set, and a cross-encoder would add latency to every document answer. It goes
  back on the table if recall drops on real corpora.
- **Unsure: citation snippets are chosen before the answer exists.** Streaming sources first
  makes chips render immediately, but the snippet is the page's best-matching passage, not
  necessarily the sentence a claim relied on. Choosing it after the answer would mean
  holding sources back until the text is done.
