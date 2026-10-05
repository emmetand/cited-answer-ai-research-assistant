import { COLLECTIONS, SEARCH_INDEXES, newId, type MemoryDoc } from '@cited/contract';
import { env } from './env.js';
import { db } from './db.js';
import { cosine, embed } from './embeddings.js';

/**
 * Long-term memory: durable facts and preferences per user, in the `memories` collection.
 * The collection is the whole truth: nothing is remembered that GET /memory does not show,
 * and deleting the row deletes the memory.
 */

const memories = async () => (await db()).collection<MemoryDoc>(COLLECTIONS.memories);

/** Injected into a prompt: at most this many memories, and roughly this many characters (~1 000 tokens). */
const RECALL_LIMIT = 10;
const RECALL_CHARS = 4000;
/** Always included, whatever the similarity: a preference is about the user, not the topic. */
const RECALL_NEWEST = 5;

export interface Recalled {
  memories: { id: string; text: string }[];
  embedTokens: number;
}

export async function saveMemory(
  userId: string,
  text: string,
  sourceThread: string,
  signal?: AbortSignal
): Promise<{ id: string; text: string; duplicate: boolean; embedTokens: number }> {
  const clean = text.trim().replace(/\s+/g, ' ').slice(0, 500);
  if (!clean) throw new Error('save_memory needs non-empty text');
  const col = await memories();
  const existing = await col.findOne({ userId, text: clean });
  if (existing) return { id: existing._id, text: clean, duplicate: true, embedTokens: 0 };

  const { vector, tokens } = await embed(clean, signal);
  const doc: MemoryDoc = {
    _id: newId('mem'),
    userId,
    text: clean,
    embedding: vector,
    sourceThread: sourceThread as MemoryDoc['sourceThread'],
    createdAt: new Date()
  };
  await col.insertOne(doc);
  return { id: doc._id, text: clean, duplicate: false, embedTokens: tokens };
}

/**
 * The memories relevant to this question: the closest by vector search, plus the newest few
 * read straight from the collection. The newest are there for two reasons: a preference
 * ("answer in British English") is never semantically close to the question it should
 * shape, and Atlas Search is eventually consistent, so a memory saved seconds ago may not
 * be in the vector index yet. The collection itself is read-your-write.
 */
export async function recallMemories(userId: string, query: string, signal?: AbortSignal): Promise<Recalled> {
  const col = await memories();
  const newest = await col
    .find({ userId }, { projection: { text: 1, createdAt: 1 } })
    .sort({ createdAt: -1 })
    .limit(RECALL_NEWEST)
    .toArray();
  if (newest.length === 0) return { memories: [], embedTokens: 0 }; // nothing saved: skip the embedding

  const { vector, tokens } = await embed(query, signal);
  let similar: { _id: string; text: string }[];
  if (env.vectorBackend === 'mongo-cosine-scan') {
    // Local mongod has no $vectorSearch: score in Node. Fine for one user's memories.
    const all = await col.find({ userId }, { projection: { text: 1, embedding: 1 } }).toArray();
    similar = all
      .map((m) => ({ _id: m._id, text: m.text, score: cosine(vector, m.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, RECALL_LIMIT);
  } else {
    similar = await col
      .aggregate<{ _id: string; text: string }>([
        {
          $vectorSearch: {
            index: SEARCH_INDEXES.memoriesVector,
            path: 'embedding',
            queryVector: vector,
            numCandidates: 100,
            limit: RECALL_LIMIT,
            filter: { userId } // inside $vectorSearch, so it filters before ranking, not after
          }
        },
        { $project: { text: 1 } }
      ])
      .toArray();
  }

  const seen = new Set<string>();
  const out: { id: string; text: string }[] = [];
  let chars = 0;
  for (const m of [...similar, ...newest]) {
    if (seen.has(m._id) || out.length >= RECALL_LIMIT || chars + m.text.length > RECALL_CHARS) continue;
    seen.add(m._id);
    chars += m.text.length;
    out.push({ id: m._id, text: m.text });
  }
  return { memories: out, embedTokens: tokens };
}

export async function listMemories(userId: string) {
  const rows = await (await memories())
    .find({ userId }, { projection: { embedding: 0 } })
    .sort({ createdAt: -1 })
    .toArray();
  return rows.map((m) => ({
    id: m._id,
    text: m.text,
    ...(m.sourceThread ? { sourceThread: m.sourceThread } : {}),
    createdAt: new Date(m.createdAt).toISOString()
  }));
}

export async function deleteMemory(userId: string, id: string): Promise<boolean> {
  const res = await (await memories()).deleteOne({ _id: id, userId });
  return res.deletedCount === 1;
}
