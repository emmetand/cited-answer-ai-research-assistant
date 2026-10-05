import type { NextFunction, Request, Response } from 'express';
import { USER_HEADER } from '@lumina/contract';
import { env } from './env.js';

/**
 * Per-user rate limit on the routes that cost money: asking and uploading. Reads (thread
 * lists, document status polling) are not limited; a UI polling an upload's progress every
 * second is not abuse.
 *
 * A sliding window in this process's memory: simple and fast, but per gateway instance and
 * lost on restart. Two gateway instances would each allow the full rate. (DESIGN.md, trade-offs.)
 */

const WINDOW_MS = 60_000;
const hits = new Map<string, number[]>();

// Forget users who have gone quiet, so the map does not grow without bound.
setInterval(() => {
  const cutoff = Date.now() - WINDOW_MS;
  for (const [user, times] of hits) if (!times.some((t) => t > cutoff)) hits.delete(user);
}, WINDOW_MS).unref();

export function rateLimit(req: Request, res: Response, next: NextFunction) {
  const user = req.header(USER_HEADER) ?? 'anonymous';
  const now = Date.now();
  const recent = (hits.get(user) ?? []).filter((t) => t > now - WINDOW_MS);

  if (recent.length >= env.rateLimitPerMinute) {
    const freesAt = recent[0]! + WINDOW_MS;
    res.setHeader('Retry-After', String(Math.ceil((freesAt - now) / 1000)));
    res.status(429).json({
      error: `rate limit: ${env.rateLimitPerMinute} questions or uploads per minute`,
      status: 429,
      resetsAt: new Date(freesAt).toISOString(),
      requestId: String(res.locals.requestId)
    });
    return;
  }
  recent.push(now);
  hits.set(user, recent);
  next();
}
