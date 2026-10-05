import Anthropic from '@anthropic-ai/sdk';
import { env, secrets } from './env.js';

/** One client per process. Retries are the SDK's (429/5xx/connection), not ours. */
export const anthropic = new Anthropic({ apiKey: secrets.anthropic, maxRetries: 2 });

/** USD per million tokens, from Anthropic's published rates. Unknown models price as Sonnet 4.6. */
const PRICES: Array<[prefix: string, input: number, output: number]> = [
  ['claude-haiku-4-5', 1, 5],
  ['claude-sonnet-5', 2, 10], // also matches claude-sonnet-5-5
  ['claude-sonnet-4-6', 3, 15],
  ['claude-opus-5-5', 4, 20],
  ['claude-opus-5', 5, 25]
];

/** What one Tavily/SerpApi call costs, matching `search_usd_per_call` in benchmark/sla.json. */
export const SEARCH_USD_PER_CALL = 0.008;

export function llmCostUsd(model: string, tokensIn: number, tokensOut: number): number {
  const [, inRate, outRate] = PRICES.find(([prefix]) => model.startsWith(prefix)) ?? ['', 3, 15];
  return (tokensIn / 1e6) * inRate + (tokensOut / 1e6) * outRate;
}

export const MODEL = env.llmModel;

/**
 * temperature: 0 makes the research turns repeatable, so a repeated question issues the
 * same searches and hits the search cache. Only older models accept sampling parameters:
 * Sonnet 5+, Opus 4.7+ and Fable reject them with a 400, so they get the model default.
 */
export const DETERMINISTIC: { temperature?: number } = /^claude-(haiku-4-5|sonnet-4-[56]|opus-4-[56])/.test(MODEL)
  ? { temperature: 0 }
  : {};
