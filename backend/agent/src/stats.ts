import { COLLECTIONS, type StatsResponse } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { deepUsedToday } from './deepCap.js';

/**
 * GET /stats, computed from the `runs` collection: the same records the run logs and the
 * agent's per-answer log lines come from, so the numbers reconcile with the logs by
 * construction rather than by a second counter that could drift.
 *
 * Service-wide, except the deep allowance, which is the asking user's own.
 */
export async function getStats(userId: string): Promise<StatsResponse> {
  const runs = (await db()).collection(COLLECTIONS.runs);
  const midnight = new Date(new Date().toISOString().slice(0, 10));

  const [totals] = await runs
    .aggregate<{ requests: number; answers: number }>([
      {
        $group: {
          _id: null,
          requests: { $sum: 1 },
          answers: { $sum: { $cond: [{ $ne: ['$terminated', 'error'] }, 1, 0] } }
        }
      }
    ])
    .toArray();

  const [today] = await runs
    .aggregate<{ cost: number; searched: number; cached: number; ttfts: number[] }>([
      { $match: { createdAt: { $gte: midnight } } },
      {
        $group: {
          _id: null,
          cost: { $sum: '$costUsd' },
          searched: { $sum: { $cond: [{ $gt: ['$webSearches', 0] }, 1, 0] } },
          cached: { $sum: { $cond: ['$searchCached', 1, 0] } },
          // TTFT is a quick-gear SLA; deep's first token comes after research by design.
          ttfts: { $push: { $cond: [{ $and: [{ $eq: ['$depth', 'quick'] }, { $isNumber: '$ttftMs' }] }, '$ttftMs', '$$REMOVE'] } }
        }
      }
    ])
    .toArray();

  return {
    requests: totals?.requests ?? 0,
    answers: totals?.answers ?? 0,
    searchCacheHitRatePct: today?.searched ? round1((100 * today.cached) / today.searched) : 0,
    ttftP95Ms: p95(today?.ttfts ?? []),
    costUsdToday: Math.round((today?.cost ?? 0) * 1e6) / 1e6,
    deepToday: await deepUsedToday(userId),
    deepDailyCap: env.deepDailyCap
  };
}

function p95(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(0.95 * sorted.length) - 1)]!;
}

const round1 = (n: number) => Math.round(n * 10) / 10;
