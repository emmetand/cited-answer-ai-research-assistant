import { createHash } from 'node:crypto';
import pino from 'pino';
import { COLLECTIONS, type SearchCacheDoc } from '@cited/contract';
import { env } from './env.js';
import { db } from './db.js';
import type { SearchResult } from './search.js';

/**
 * The search cache, two tiers:
 *   1. an in-process LRU (fast, lost on restart, per instance)
 *   2. the `searchCache` collection, expired by its TTL index on `expiresAt` (shared, survives restarts)
 *
 * It is a cache, never a source of truth: if Mongo is unreachable the search just goes to
 * the provider. Failing open here is correct — a miss costs money, not honesty.
 */

const log = pino({ level: env.logLevel });
const LRU_MAX = 500;

type Entry = { results: SearchResult[]; expiresAt: number };
const lru = new Map<string, Entry>();

const STOPWORDS = new Set(
  'a an and are as at be by does do for from how in is it of on or that the this to what when where which who why with vs versus'.split(' ')
);

/**
 * Normalized as a bag of words: lowercase, punctuation and filler words dropped, the rest
 * deduplicated and sorted. "TTL index MongoDB" and "what is a mongodb ttl index?" share a key,
 * because they return the same pages, and a repeated question rarely reproduces its searches
 * word for word.
 */
export function normalizeQuery(query: string): string {
  const words = query
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w && !STOPWORDS.has(w));
  return [...new Set(words)].sort().join(' ');
}

export const cacheKey = (query: string, provider: string) =>
  createHash('sha256').update(`${provider}\n${normalizeQuery(query)}`).digest('hex');

/** "latest", "today", or a year from this one on: the answer changes, so the cache is skipped. */
export function isTimeSensitive(query: string): boolean {
  if (/\b(today|tonight|yesterday|latest|newest|current|currently|now|recent|recently|breaking|this (week|month|year))\b/i.test(query)) {
    return true;
  }
  const thisYear = new Date().getFullYear();
  return [...query.matchAll(/\b(19|20)\d{2}\b/g)].some((m) => Number(m[0]) >= thisYear);
}

export async function cacheGet(key: string): Promise<SearchResult[] | null> {
  const now = Date.now();
  const hit = lru.get(key);
  if (hit) {
    lru.delete(key);
    if (hit.expiresAt > now) {
      lru.set(key, hit); // most recently used goes to the back
      return hit.results;
    }
  }
  try {
    // The TTL monitor only sweeps once a minute, so expiry is also checked on read.
    const doc = await (await db())
      .collection<SearchCacheDoc>(COLLECTIONS.searchCache)
      .findOne({ _id: key, expiresAt: { $gt: new Date(now) } });
    if (!doc) return null;
    const results = doc.results as unknown as SearchResult[];
    remember(key, { results, expiresAt: new Date(doc.expiresAt).getTime() });
    return results;
  } catch (err) {
    log.warn({ err: (err as Error).message }, 'searchCache read failed; treating as a miss');
    return null;
  }
}

export async function cachePut(key: string, query: string, provider: 'tavily' | 'serpapi', results: SearchResult[]) {
  const now = Date.now();
  const expiresAt = now + env.searchCacheTtlSeconds * 1000;
  remember(key, { results, expiresAt });
  try {
    await (await db())
      .collection<SearchCacheDoc>(COLLECTIONS.searchCache)
      .replaceOne(
        { _id: key },
        { provider, query, results: results as unknown as Record<string, unknown>[], expiresAt: new Date(expiresAt), createdAt: new Date(now) },
        { upsert: true }
      );
  } catch (err) {
    log.warn({ err: (err as Error).message }, 'searchCache write failed; the LRU still has it');
  }
}

function remember(key: string, entry: Entry) {
  lru.delete(key);
  lru.set(key, entry);
  if (lru.size > LRU_MAX) lru.delete(lru.keys().next().value!);
}
