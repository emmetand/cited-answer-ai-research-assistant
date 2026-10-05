import type { Request, Response } from 'express';
import pino from 'pino';
import { REQUEST_HEADER, USER_HEADER } from '@cited/contract';
import { env } from './env.js';

const log = pino({ level: env.logLevel });

export interface ForwardOptions {
  /** A validated JSON body to send instead of the original. */
  body?: unknown;
  /** Stream the original request body through untouched (multipart uploads). */
  raw?: boolean;
  /** Give up on the agent after this long. Omitted for the ask stream, which the agent bounds itself. */
  timeoutMs?: number;
}

/**
 * Forward one request to the agent service and relay its answer: status, body, and the
 * request id. Three rules:
 *
 *   - Streams pass straight through, chunk by chunk, never buffered. Nothing in this
 *     server compresses, and each chunk is written the moment it arrives.
 *   - The browser leaving cancels the upstream request, which the agent sees as its
 *     client going away and stops spending on.
 *   - The agent being unreachable is a 502, never a 2xx with an empty body.
 */
export async function forward(req: Request, res: Response, opts: ForwardOptions = {}): Promise<void> {
  const requestId = String(res.locals.requestId);
  const upstreamAbort = new AbortController();
  let clientGone = false;
  res.on('close', () => {
    if (!res.writableFinished) {
      clientGone = true;
      upstreamAbort.abort();
    }
  });

  const headers: Record<string, string> = { [REQUEST_HEADER]: requestId };
  const user = req.header(USER_HEADER);
  if (user) headers[USER_HEADER] = user;

  let body: unknown;
  if (opts.raw) {
    for (const h of ['content-type', 'content-length']) {
      const v = req.header(h);
      if (v) headers[h] = v;
    }
    body = req; // the untouched multipart stream
  } else if (opts.body !== undefined) {
    headers['content-type'] = 'application/json';
    body = JSON.stringify(opts.body);
  }

  const signal = opts.timeoutMs
    ? AbortSignal.any([upstreamAbort.signal, AbortSignal.timeout(opts.timeoutMs)])
    : upstreamAbort.signal;

  let upstream: globalThis.Response;
  try {
    upstream = await fetch(`${env.agentUrl}${req.originalUrl}`, {
      method: req.method,
      headers,
      body: body as RequestInit['body'],
      signal,
      // Required by Node's fetch to send a streaming request body.
      ...(opts.raw ? { duplex: 'half' } : {})
    } as RequestInit);
  } catch (err) {
    if (clientGone) return;
    const reason = (err as Error).name === 'TimeoutError' ? `timed out after ${opts.timeoutMs} ms` : (err as Error).message;
    log.error({ requestId, route: req.originalUrl, err: reason }, 'agent unreachable');
    res.status(502).json({ error: `agent service unavailable: ${reason}`, status: 502, requestId });
    return;
  }

  res.status(upstream.status);
  for (const h of ['content-type', 'retry-after']) {
    const v = upstream.headers.get(h);
    if (v) res.setHeader(h, v);
  }
  if (!upstream.body || upstream.status === 204) {
    res.end();
    return;
  }

  const isStream = upstream.headers.get('content-type')?.includes('text/event-stream') ?? false;
  if (isStream) {
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no'); // proxies in front of us must not buffer either
    res.flushHeaders();
    res.socket?.setNoDelay(true);
  }

  try {
    for await (const chunk of upstream.body as unknown as AsyncIterable<Uint8Array>) {
      if (clientGone) break;
      res.write(chunk);
    }
    res.end();
  } catch (err) {
    if (clientGone) return;
    // The agent died mid-response. On a stream the status line is long gone, so the
    // failure is reported in-band, as the contract's error event.
    log.error({ requestId, route: req.originalUrl, err: (err as Error).message }, 'agent stream broke');
    if (isStream) res.write(`event: error\ndata: ${JSON.stringify({ status: 502, error: 'the agent service stopped mid-answer' })}\n\n`);
    res.end();
  }
}
