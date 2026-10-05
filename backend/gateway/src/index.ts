/**
 * LUMINA gateway — the software backend. PROVIDED SKELETON: YOU BUILD THIS OUT.
 *
 * What is already here: the server, CORS, the request id, the pino request log, /health
 * (which nests the agent service's health), a 501 for every contract route, and the
 * static hosting of web/dist. That is deliberately the boring half.
 *
 * What you build (backend/gateway/, see README Part 2):
 *   1. X-User-Id enforcement           → 401 without it, on every route but /health
 *   2. zod validation from @lumina/contract → 400 on a bad body, with the zod message
 *   3. a per-user rate limit           → 429
 *   4. the proxy to the agent service, and SSE pass-through for /threads/:id/ask
 *   5. 502 for any upstream failure    → never a 2xx when the agent threw
 *
 * The browser talks ONLY to this service. No provider key is ever read here.
 */
import express from 'express';
import cors from 'cors';
import { pinoHttp } from 'pino-http';
import pino from 'pino';
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import type { ZodTypeAny } from 'zod';
import {
  AskBody,
  CreateSpaceBody,
  CreateThreadBody,
  HealthResponse,
  MAX_UPLOAD_BYTES,
  REQUEST_HEADER,
  USER_HEADER
} from '@lumina/contract';
import { env } from './env.js';
import { forward } from './proxy.js';
import { rateLimit } from './rateLimit.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use(cors({ origin: env.corsOrigins, credentials: false, exposedHeaders: [REQUEST_HEADER] }));

// One request id, reused if the caller sent one, generated if not, forwarded to the agent
// service and logged by both. This is what makes one request greppable end to end.
app.use((req, res, next) => {
  const id = (req.header(REQUEST_HEADER) ?? `req_${randomUUID().slice(0, 12)}`).trim();
  res.locals.requestId = id;
  res.setHeader(REQUEST_HEADER, id);
  next();
});

app.use(
  pinoHttp({
    logger: log,
    genReqId: (_req, res) => String(res.locals.requestId),
    customProps: (req, res) => ({
      requestId: res.locals.requestId,
      userId: req.header(USER_HEADER) ?? null
    }),
    // The ask route is a stream; one line when it closes is the useful line.
    autoLogging: true
  })
);

// JSON everywhere except the multipart upload route, which your handler owns.
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

// ---------------------------------------------------------------- /health (implemented)

app.get('/health', async (_req, res) => {
  let ai: { status: 'ok' | 'down' } & Record<string, unknown> = { status: 'down' };
  try {
    const upstream = await fetch(`${env.agentUrl}/health`, { signal: AbortSignal.timeout(3000) });
    const body = (await upstream.json()) as Record<string, unknown>;
    ai = { ...body, status: upstream.ok ? 'ok' : 'down' };
  } catch (err) {
    // Health tells the truth about a dead dependency. It never pretends.
    ai = { status: 'down', error: (err as Error).message };
  }

  const body: HealthResponse = {
    status: ai.status === 'ok' ? 'ok' : 'degraded',
    model: String(ai.model ?? 'unset'),
    searchProvider: (ai.searchProvider as HealthResponse['searchProvider']) ?? 'tavily',
    vectorStore: (ai.vectorStore as HealthResponse['vectorStore']) ?? 'atlas-vector-search',
    db: (ai.db as HealthResponse['db']) ?? 'down',
    ai
  };
  res.status(ai.status === 'ok' ? 200 : 503).json(body);
});

// ---------------------------------------------------------------- the evaluation page's data

// Public, like the page that renders it: a stranger opening /evals has no X-User-Id.
app.get('/evals/report.json', (_req, res) => {
  if (!existsSync(env.reportPath)) {
    return res.status(404).json({
      error: 'no evaluation report yet: run /fde-lumina-eval to produce reports/report.json',
      status: 404,
      requestId: String(res.locals.requestId)
    });
  }
  res.sendFile(env.reportPath);
});

// ---------------------------------------------------------------- identity

// Every API route but /health and the report needs X-User-Id. The static UI does not: a
// browser loading the page has not chosen an id yet.
const API_ROUTE = /^\/(stats|threads|memory|spaces)(\/|$)/;
app.use((req, res, next) => {
  if (!API_ROUTE.test(req.path) || req.header(USER_HEADER)?.trim()) return next();
  res.status(401).json({ error: 'X-User-Id header is required', status: 401, requestId: String(res.locals.requestId) });
});

// ---------------------------------------------------------------- validation

/** Reject a malformed body here, at the edge, with the contract's own message. */
const validate =
  (schema: ZodTypeAny) => (req: express.Request, res: express.Response, next: express.NextFunction) => {
    const parsed = schema.safeParse(req.body ?? {});
    if (!parsed.success) {
      const issue = parsed.error.issues[0];
      return res.status(400).json({
        error: issue ? `${issue.path.length ? `${issue.path.join('.')}: ` : ''}${issue.message}` : 'invalid body',
        status: 400,
        requestId: String(res.locals.requestId)
      });
    }
    res.locals.body = parsed.data; // forwarded with the contract's defaults filled in
    next();
  };

// ---------------------------------------------------------------- the API: proxied to the agent

const SHORT = { timeoutMs: 30_000 };
const json = (req: express.Request, res: express.Response) => forward(req, res, { ...SHORT, body: res.locals.body });

app.get('/stats', (req, res) => forward(req, res, SHORT));

app.post('/threads', validate(CreateThreadBody), json);
app.get('/threads', (req, res) => forward(req, res, SHORT));
app.get('/threads/:threadId', (req, res) => forward(req, res, SHORT));
// The answer stream: rate-limited, validated, then passed through unbuffered. No timeout
// here: the agent bounds every run with its own per-gear wall-clock cap.
app.post('/threads/:threadId/ask', rateLimit, validate(AskBody), (req, res) =>
  forward(req, res, { body: res.locals.body })
);

app.get('/memory', (req, res) => forward(req, res, SHORT));
app.delete('/memory/:memoryId', (req, res) => forward(req, res, SHORT));

app.post('/spaces', validate(CreateSpaceBody), json);
app.get('/spaces', (req, res) => forward(req, res, SHORT));
app.post('/spaces/:spaceId/documents', rateLimit, (req, res) => {
  // Refuse an oversized upload before streaming 25 MB+ across to the agent.
  if (Number(req.header('content-length') ?? 0) > MAX_UPLOAD_BYTES + 64 * 1024) {
    return res.status(413).json({
      error: `file is larger than ${MAX_UPLOAD_BYTES / 1024 / 1024} MB`,
      status: 413,
      requestId: String(res.locals.requestId)
    });
  }
  return forward(req, res, { raw: true, timeoutMs: 120_000 });
});
app.get('/spaces/:spaceId/documents', (req, res) => forward(req, res, SHORT));

// ---------------------------------------------------------------- static UI

// In production the gateway serves the built UI, so / and /evals come from one origin.
if (existsSync(env.webDist)) {
  app.use(express.static(env.webDist));
  app.get(/^(?!\/(health|stats|threads|memory|spaces|artifacts|evals)).*/, (_req, res) => {
    res.sendFile(`${env.webDist}/index.html`);
  });
}

app.use((req, res) => {
  res.status(404).json({ error: `no route ${req.method} ${req.path}`, status: 404 });
});

// A thrown error is a 502 with a log line, never a 200 with a plausible body (rule A1).
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  log.error({ err, requestId: res.locals.requestId }, 'gateway error');
  res.status(502).json({ error: err.message, status: 502, requestId: String(res.locals.requestId) });
});

app.listen(env.port, () => {
  log.info(
    { port: env.port, agentUrl: env.agentUrl, cors: env.corsOrigins },
    'gateway up'
  );
});
