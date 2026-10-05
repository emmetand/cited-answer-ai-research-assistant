import OpenAI from 'openai';
import { EMBEDDING_DIMS } from '@lumina/contract';
import { env, secrets } from './env.js';

/** Embeddings come from OpenAI even though answers come from Claude: the assignment names text-embedding-3-small. */
const openai = new OpenAI({ apiKey: secrets.openai, maxRetries: 2, timeout: 15_000 });

/** USD per million embedding tokens, matching `embedding_usd_per_mtok` in benchmark/sla.json. */
export const EMBED_USD_PER_MTOK = 0.02;

export async function embed(text: string, signal?: AbortSignal): Promise<{ vector: number[]; tokens: number }> {
  if (!secrets.openai) throw new Error('OPENAI_API_KEY is not set');
  const res = await openai.embeddings.create({ model: env.embeddingModel, input: text }, { signal });
  const vector = res.data[0]?.embedding;
  if (!vector || vector.length !== EMBEDDING_DIMS) {
    throw new Error(`embedding has ${vector?.length ?? 0} dims, the index expects ${EMBEDDING_DIMS}`);
  }
  return { vector, tokens: res.usage.total_tokens };
}

/** Embed many texts in batched requests. Order is preserved. */
export async function embedMany(texts: string[], batchSize = 64): Promise<{ vectors: number[][]; tokens: number }> {
  if (!secrets.openai) throw new Error('OPENAI_API_KEY is not set');
  const vectors: number[][] = [];
  let tokens = 0;
  for (let i = 0; i < texts.length; i += batchSize) {
    const res = await openai.embeddings.create({ model: env.embeddingModel, input: texts.slice(i, i + batchSize) });
    for (const d of [...res.data].sort((a, b) => a.index - b.index)) {
      if (d.embedding.length !== EMBEDDING_DIMS) throw new Error(`embedding has ${d.embedding.length} dims, expected ${EMBEDDING_DIMS}`);
      vectors.push(d.embedding);
    }
    tokens += res.usage.total_tokens;
  }
  return { vectors, tokens };
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}
