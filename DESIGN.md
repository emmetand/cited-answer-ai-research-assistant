# DESIGN.md — LUMINA

## Components

- **UI:** the provided React app, deployed on Vercel. Unmodified.
- **Gateway:** a small Express server, public on Fly.io. It's the only thing the browser talks to.
- **Agent service:** an Express server on Fly.io, private (only the gateway can reach it). It runs the agent loop: ask Claude, call a tool, look at the result, repeat, then write the answer.
- **Jobs worker:** runs as a separate process next to the agent service. It picks up uploaded documents, splits them into chunks, embeds them, and saves them. It's kept separate so a big PDF doesn't slow down answers being streamed.
- **MongoDB Atlas (free tier):** one database for everything: chats, memories, documents, chunks with their vectors, the cache, the job queue, and logs.
- **Outside services:** Tavily for search, Claude for answers, OpenAI for embeddings.
- **Search cache:** a small in-memory cache inside the agent, backed by a Mongo collection whose entries expire after 6 hours.
- **Run logs:** one JSON file per answer, recording every step. The grader reads these.

## Responsibilities

- **Gateway:** checks the user ID, assigns a request ID, logs requests, rejects malformed input, rate-limits, and passes the stream through. It never holds an API key and never calls an AI provider.
- **Agent:** the only component holding API keys. It runs the loop, picks tools, enforces the tool-call and time caps, and enforces the deep-search daily cap. The cap lives here because if it were in the gateway, someone could bypass it by calling the agent directly.
- **Worker:** the only component that turns an uploaded file into searchable chunks. It doesn't answer questions.
- **Quick vs. deep:** only the user picks deep. Quick searches are simply never given the `plan_research` tool, so they can't escalate on their own.

## Communication

- **Browser ↔ gateway:** normal HTTP requests, plus a stream for answers (SSE, server-sent events: the server pushes words as they're written).
- **Gateway ↔ agent:** the same thing; the gateway just forwards it.
- **Agent ↔ worker:** they never talk directly. The agent writes a "to-do" row in the `jobs` collection, and the worker checks that collection for new rows. Why: if the worker is down, uploads still succeed and simply wait in the queue.

When things break:

- **Agent down:** the gateway returns `502`.
- **Claude or Tavily fails mid-answer:** send an `error` event, mark the run `terminated: "error"`, and never make up a plausible answer.
- **User closes the tab:** stop the loop so it stops spending money.
- **Worker crashes mid-job:** the job stays marked "running". A cleanup step later notices it's stale and puts it back in the queue.

## State

- **Precious (the source of truth):** threads and messages, memories, spaces, document records, and the original uploaded files (stored in GridFS, MongoDB's built-in file storage).
- **Throwaway (caches):** the in-memory search cache and the `searchCache` collection. Deleting them only makes things slower.
- **Somewhere in between:** `chunks`. They can be rebuilt from the original files, but rebuilding costs time and embedding money.
- **Written but not yet searchable:** after the worker saves the chunks, the document stays at `embedding` until a test search actually finds them. Only then does it switch to `indexed`. Why: Atlas updates its search index a few seconds after the write, not instantly.
- **Weak spot:** rate-limit counters kept in memory reset whenever the server restarts.

## Trade-offs

- **Atlas for vectors instead of a dedicated vector database:** everything sits in one place, but the free tier allows only 3 search indexes. I hit this during setup and had to delete Atlas's sample data to make room.
- **Two services instead of one:** keys are safer and failures are isolated, but there's more to deploy and one extra network hop.
- **Haiku while building, Sonnet for grading:** cheaper development, but true quality and speed only show up late in the process.
- **Rate limits in memory instead of in the database:** simple and fast, but they're lost on restart and not shared between servers.
