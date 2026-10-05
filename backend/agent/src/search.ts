import { env, secrets } from './env.js';
import { cacheGet, cacheKey, cachePut, isTimeSensitive } from './searchCache.js';

export interface SearchResult {
  title: string;
  url: string;
  /** The provider's snippet. Used to choose what to fetch, never as a citation's evidence. */
  content: string;
  /** The provider's relevance score, 0–1, when it gives one (Tavily does; SerpApi does not). */
  score?: number;
}

export interface SearchOutcome {
  results: SearchResult[];
  /** True when served from the search cache, with no provider call. */
  cached: boolean;
  /** True when the cache was skipped because the question is time-sensitive. */
  bypassed: boolean;
}

const TIMEOUT_MS = 10_000;

/**
 * Web search through whichever provider SEARCH_PROVIDER names, behind the two-tier cache.
 * Throws on any provider failure: the caller turns that into a visible failed trace step,
 * never an empty result. Failures are never cached.
 *
 * `fresh` skips the cache for this call (the user's question asked for something current);
 * a query that is itself time-sensitive skips it too.
 */
export async function webSearch(query: string, signal: AbortSignal, fresh = false): Promise<SearchOutcome> {
  const provider = env.searchProvider;
  const bypassed = fresh || isTimeSensitive(query);
  const key = cacheKey(query, provider);

  if (!bypassed) {
    const hit = await cacheGet(key);
    if (hit) return { results: hit, cached: true, bypassed };
  }

  const timeout = AbortSignal.any([signal, AbortSignal.timeout(TIMEOUT_MS)]);
  const results = provider === 'serpapi' ? await serpapi(query, timeout) : await tavily(query, timeout);
  if (!bypassed) await cachePut(key, query, provider, results);
  return { results, cached: false, bypassed };
}

async function tavily(query: string, signal: AbortSignal): Promise<SearchResult[]> {
  if (!secrets.tavily) throw new Error('TAVILY_API_KEY is not set');
  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${secrets.tavily}` },
    body: JSON.stringify({ query, max_results: 5, search_depth: 'basic' }),
    signal
  });
  if (!res.ok) throw new Error(`tavily ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const body = (await res.json()) as { results?: Array<{ title?: string; url?: string; content?: string; score?: number }> };
  return (body.results ?? [])
    .filter((r) => r.url)
    .map((r) => ({
      title: r.title || r.url!,
      url: r.url!,
      content: r.content ?? '',
      ...(typeof r.score === 'number' ? { score: r.score } : {})
    }));
}

async function serpapi(query: string, signal: AbortSignal): Promise<SearchResult[]> {
  if (!secrets.serpapi) throw new Error('SERPAPI_API_KEY is not set');
  const url = new URL('https://serpapi.com/search.json');
  url.searchParams.set('engine', 'google');
  url.searchParams.set('q', query);
  url.searchParams.set('num', '5');
  url.searchParams.set('api_key', secrets.serpapi);
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`serpapi ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const body = (await res.json()) as { organic_results?: Array<{ title?: string; link?: string; snippet?: string }> };
  return (body.organic_results ?? [])
    .filter((r) => r.link)
    .slice(0, 5)
    .map((r) => ({ title: r.title || r.link!, url: r.link!, content: r.snippet ?? '' }));
}
