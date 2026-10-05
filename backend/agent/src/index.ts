/**
 * Cited agent service — where the work happens. Provider keys live only in this process,
 * and only the gateway may reach it.
 *
 *   POST /threads/:id/ask   the answer stream: trace → sources → token → done
 *                           (deep: plan first), quick or deep gear (loop.ts, deep.ts)
 *   /threads, /memory       conversation history and long-term memory
 *   /spaces/...             document upload (202 + jobs queue) and status
 *   GET /stats, /health     service-wide numbers, and what is live
 *
 * Three rules every route keeps:
 *   - Fail loud. A provider exception ends the run with terminated:"error" and a 502.
 *     Never a try/catch that returns a plausible answer: a swallowed exception turns a
 *     broken dependency into confident, wrong output that nothing alerts on.
 *   - Grounded or nothing. Every [n] resolves to something retrieved in THIS request.
 *   - Depth is opted into, never drifted into. A quick search cannot call plan_research,
 *     however much the model would like to. A product that escalates its own spend is a
 *     product with an unbounded bill.
 */
import express from 'express';
import multer from 'multer';
import pino from 'pino';
import { mkdirSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import {
  AskBody,
  COLLECTIONS,
  CreateSpaceBody,
  CreateThreadBody,
  HealthResponse,
  MAX_UPLOAD_BYTES,
  REQUEST_HEADER,
  USER_HEADER,
  newId,
  type MessageDoc,
  type ThreadDoc
} from '@cited/contract';
import { env } from './env.js';
import { db, pingDb } from './db.js';
import { SseStream } from './sse.js';
import { runQuickAsk, type HistoryTurn } from './loop.js';
import { runDeepAsk } from './deep.js';
import { resetsAt, takeDeepSlot } from './deepCap.js';
import { getStats } from './stats.js';
import { deleteMemory, listMemories } from './memory.js';
import {
  createSpace,
  deleteFile,
  findSpace,
  listDocuments,
  mimeFor,
  newDocId,
  registerUpload,
  spaces,
  storeFile
} from './documents.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

mkdirSync(env.runsDir, { recursive: true });

// ---------------------------------------------------------------- /health (implemented)

app.get('/health', async (_req, res) => {
  const dbStatus = await pingDb();
  const body: HealthResponse = {
    status: dbStatus === 'ok' ? 'ok' : 'degraded',
    model: env.llmModel,
    searchProvider: env.searchProvider,
    vectorStore: env.vectorBackend,
    db: dbStatus,
    ai: { status: 'ok' }
  };
  res.status(dbStatus === 'ok' ? 200 : 503).json(body);
});

// ---------------------------------------------------------------- identity

// The gateway enforces X-User-Id too, but the agent does not trust that it was reached
// through the gateway: every user-scoped route checks it here as well.
app.use((req, res, next) => {
  const requestId = (req.header(REQUEST_HEADER) ?? `req_${randomUUID().slice(0, 12)}`).trim();
  res.locals.requestId = requestId;
  res.setHeader(REQUEST_HEADER, requestId);
  if (req.path === '/health') return next();
  const userId = req.header(USER_HEADER)?.trim();
  if (!userId) return res.status(401).json({ error: 'X-User-Id header is required', status: 401, requestId });
  res.locals.userId = userId;
  next();
});

// ---------------------------------------------------------------- threads

const threads = async () => (await db()).collection<ThreadDoc>(COLLECTIONS.threads);
const messages = async () => (await db()).collection<MessageDoc>(COLLECTIONS.messages);

const iso = (d: Date | string) => new Date(d).toISOString();

/** Turns the model sees as conversation context: the last few, bounded, citations removed. */
const HISTORY_MESSAGES = 6;
const HISTORY_CHARS = 1500;

async function loadHistory(threadId: string): Promise<HistoryTurn[]> {
  const recent = await (await messages())
    .find({ threadId })
    .sort({ createdAt: -1 })
    .limit(HISTORY_MESSAGES)
    .toArray();
  const turns = recent.reverse().map((m) => ({
    role: m.role,
    // An old [n] points at an old request's sources; left in, the model might reuse it.
    content: m.content.replace(/\[\d{1,3}\]/g, '').slice(0, HISTORY_CHARS)
  }));
  while (turns[0]?.role === 'assistant') turns.shift(); // a conversation must open with the user
  return turns;
}

app.get('/threads', async (_req, res, next) => {
  try {
    const rows = await (await threads())
      .find({ userId: res.locals.userId })
      .sort({ createdAt: -1 })
      .limit(50)
      .toArray();
    res.json({ threads: rows.map((t) => ({ threadId: t._id, title: t.title, createdAt: iso(t.createdAt) })) });
  } catch (err) {
    next(err);
  }
});

app.get('/threads/:threadId', async (req, res, next) => {
  try {
    const thread = await (await threads()).findOne({ _id: req.params.threadId, userId: res.locals.userId });
    if (!thread) return res.status(404).json({ error: `no thread ${req.params.threadId}`, status: 404 });
    const rows = await (await messages()).find({ threadId: thread._id }).sort({ createdAt: 1 }).toArray();
    res.json({
      threadId: thread._id,
      title: thread.title,
      messages: rows.map((m) => ({
        role: m.role,
        content: m.content,
        ...(m.role === 'assistant' ? { sources: m.sources ?? [] } : {}),
        ...(m.answerId ? { answerId: m.answerId } : {}),
        ...(m.done ? { done: m.done } : {}),
        createdAt: iso(m.createdAt)
      }))
    });
  } catch (err) {
    next(err);
  }
});

app.post('/threads', async (req, res, next) => {
  try {
    const body = CreateThreadBody.safeParse(req.body ?? {});
    if (!body.success) return res.status(400).json({ error: body.error.issues[0]?.message ?? 'invalid body', status: 400 });
    const thread: ThreadDoc = {
      _id: newId('thr'),
      userId: res.locals.userId,
      title: body.data.title ?? 'New thread',
      createdAt: new Date()
    };
    await (await threads()).insertOne(thread);
    res.status(201).json({ threadId: thread._id });
  } catch (err) {
    next(err);
  }
});

// ---------------------------------------------------------------- stats

app.get('/stats', async (_req, res, next) => {
  try {
    res.json(await getStats(res.locals.userId));
  } catch (err) {
    next(err);
  }
});

// ---------------------------------------------------------------- memory

app.get('/memory', async (_req, res, next) => {
  try {
    res.json({ memories: await listMemories(res.locals.userId) });
  } catch (err) {
    next(err);
  }
});

app.delete('/memory/:memoryId', async (req, res, next) => {
  try {
    const gone = await deleteMemory(res.locals.userId, req.params.memoryId);
    if (!gone) return res.status(404).json({ error: `no memory ${req.params.memoryId}`, status: 404 });
    res.status(204).end();
  } catch (err) {
    next(err);
  }
});

// ---------------------------------------------------------------- spaces & documents

app.post('/spaces', async (req, res, next) => {
  try {
    const body = CreateSpaceBody.safeParse(req.body ?? {});
    if (!body.success) return res.status(400).json({ error: body.error.issues[0]?.message ?? 'invalid body', status: 400 });
    const space = await createSpace(res.locals.userId, body.data.name);
    res.status(201).json({ spaceId: space._id, name: space.name });
  } catch (err) {
    next(err);
  }
});

app.get('/spaces', async (_req, res, next) => {
  try {
    const rows = await (await spaces()).find({ userId: res.locals.userId }).sort({ createdAt: -1 }).toArray();
    res.json({ spaces: rows.map((s) => ({ spaceId: s._id, name: s.name, createdAt: iso(s.createdAt) })) });
  } catch (err) {
    next(err);
  }
});

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_UPLOAD_BYTES, files: 1 } }).single('file');

app.post('/spaces/:spaceId/documents', (req, res, next) => {
  const t0 = Date.now();
  upload(req, res, async (uploadErr: unknown) => {
    try {
      if (uploadErr instanceof multer.MulterError) {
        const tooBig = uploadErr.code === 'LIMIT_FILE_SIZE';
        return res.status(tooBig ? 413 : 400).json({
          error: tooBig ? `file is larger than ${MAX_UPLOAD_BYTES / 1024 / 1024} MB` : uploadErr.message,
          status: tooBig ? 413 : 400
        });
      }
      if (uploadErr) throw uploadErr;
      if (!req.file) return res.status(400).json({ error: 'multipart field "file" is required', status: 400 });
      const mime = mimeFor(req.file.originalname, req.file.mimetype);
      if (!mime) return res.status(400).json({ error: 'only PDF, Markdown (.md) and plain text (.txt) are accepted', status: 400 });

      // The Space lookup and the GridFS write overlap: one round trip to Atlas fewer on
      // every upload, at the cost of deleting the file on the rare request for a bad Space.
      const userId: string = res.locals.userId;
      const docId = newDocId();
      const [space, fileId] = await Promise.all([
        findSpace(userId, req.params.spaceId),
        storeFile(req.file.originalname, req.file.buffer, { docId, spaceId: req.params.spaceId, userId, mimeType: mime })
      ]);
      if (!space) {
        await deleteFile(fileId);
        return res.status(404).json({ error: `no space ${req.params.spaceId}`, status: 404 });
      }
      await registerUpload(docId, fileId, userId, space._id, req.file.originalname, mime, req.file.size);
      log.info({ requestId: res.locals.requestId, docId, bytes: req.file.size, ms: Date.now() - t0 }, 'upload accepted');
      res.status(202).json({ docId, status: 'pending' });
    } catch (err) {
      next(err);
    }
  });
});

app.get('/spaces/:spaceId/documents', async (req, res, next) => {
  try {
    const space = await findSpace(res.locals.userId, req.params.spaceId);
    if (!space) return res.status(404).json({ error: `no space ${req.params.spaceId}`, status: 404 });
    res.json({ documents: await listDocuments(space._id) });
  } catch (err) {
    next(err);
  }
});

// ---------------------------------------------------------------- ask (quick gear)

app.post('/threads/:threadId/ask', async (req, res, next) => {
  const requestId: string = res.locals.requestId;
  const userId: string = res.locals.userId;
  try {
    const body = AskBody.safeParse(req.body ?? {});
    if (!body.success) {
      return res.status(400).json({ error: body.error.issues[0]?.message ?? 'invalid body', status: 400, requestId });
    }
    const { query, mode, depth, spaceId } = body.data;
    if (mode === 'docs' && !spaceId) {
      return res.status(400).json({ error: 'mode "docs" needs a spaceId: which documents should it search?', status: 400, requestId });
    }

    const [thread, space] = await Promise.all([
      (await threads()).findOne({ _id: req.params.threadId, userId }),
      spaceId ? findSpace(userId, spaceId) : Promise.resolve(null)
    ]);
    if (!thread) return res.status(404).json({ error: `no thread ${req.params.threadId}`, status: 404, requestId });
    if (spaceId && !space) return res.status(404).json({ error: `no space ${spaceId}`, status: 404, requestId });

    // The spend gate, enforced here next to the spending, before anything streams. Quick is
    // the default and is never upgraded: only an explicit depth: "deep" reaches this.
    if (depth === 'deep' && !(await takeDeepSlot(userId))) {
      return res.status(429).json({
        error: `daily deep-search limit reached (${env.deepDailyCap} per day); quick search is still available`,
        status: 429,
        resetsAt: resetsAt(),
        requestId
      });
    }

    // If the user closes the tab, stop the loop: an answer nobody reads still costs money.
    const clientGone = new AbortController();
    res.on('close', () => {
      if (!res.writableFinished) clientGone.abort();
    });

    // History is read BEFORE this question is saved, so it holds only earlier turns.
    const history = await loadHistory(thread._id);
    await (await messages()).insertOne({
      _id: `msg_${randomUUID()}`,
      threadId: thread._id,
      userId,
      role: 'user',
      content: query,
      sources: [],
      createdAt: new Date()
    });
    if (thread.title === 'New thread') {
      await (await threads()).updateOne({ _id: thread._id }, { $set: { title: query.slice(0, 80) } });
    }

    const sse = new SseStream(res);
    try {
      const askInput = { query, requestId, userId, threadId: thread._id, history, mode, spaceId: space?._id };
      const s =
        depth === 'deep'
          ? await runDeepAsk(askInput, sse, clientGone.signal)
          : await runQuickAsk(askInput, sse, clientGone.signal);
      // Only a finished stream is saved as an answer. An error saves nothing pretending to be one.
      if (s.done) {
        await (await messages()).insertOne({
          _id: `msg_${randomUUID()}`,
          threadId: thread._id,
          userId,
          role: 'assistant',
          content: s.answerText,
          answerId: s.answerId,
          sources: s.sources,
          done: s.done,
          ...(s.plan?.length ? { subQuestions: s.plan } : {}),
          createdAt: new Date()
        });
      }
      log.info(
        {
          requestId,
          toolCalls: s.toolCalls,
          terminated: s.terminated,
          tokens: s.tokens,
          costUsd: s.costUsd,
          searchCached: s.searchCached,
          ttftMs: s.ttftMs,
          latencyMs: s.latencyMs,
          depth,
          ...(s.plan?.length ? { subQuestions: s.plan.length } : {}),
          ...(s.danglingCitations.length ? { danglingCitations: s.danglingCitations } : {})
        },
        clientGone.signal.aborted ? 'answer abandoned: client disconnected' : 'answer'
      );
    } catch (err) {
      // Fail loud: a provider exception is a 502, never a plausible answer.
      const error = err instanceof Error ? err.message : String(err);
      log.error({ requestId, err: error, terminated: 'error' }, 'answer failed');
      if (sse.isOpen) sse.send('error', { status: 502, error });
      else res.status(502).json({ error, status: 502, requestId });
    } finally {
      sse.end();
    }
  } catch (err) {
    next(err);
  }
});

app.use((req, res) => res.status(404).json({ error: `no route ${req.method} ${req.path}`, status: 404 }));

app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  log.error({ err }, 'agent error');
  res.status(502).json({ error: err.message, status: 502 });
});

app.listen(env.port, () => {
  log.info(
    {
      port: env.port,
      model: env.llmModel,
      searchProvider: env.searchProvider,
      vectorStore: env.vectorBackend,
      caps: {
        quick: { toolCalls: env.maxToolCalls, wallClockSec: env.maxWallClockSec },
        deep: { toolCalls: env.maxToolCallsDeep, wallClockSec: env.maxWallClockSecDeep, dailyCap: env.deepDailyCap }
      }
    },
    'agent up'
  );
});
