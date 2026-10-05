import { SEARCH_INDEXES, type ChunkDoc, type DocumentDoc, type Locator } from '@lumina/contract';
import { env } from './env.js';
import { chunks, documents } from './documents.js';
import { cosine, embed } from './embeddings.js';

/**
 * Hybrid retrieval over one Space: dense ($vectorSearch, meaning) and lexical ($search /
 * BM25, exact words) run in parallel, each filtered to the Space INSIDE the search stage,
 * then fused with reciprocal rank fusion.
 *
 * Why both: dense finds "automobile" for "car"; BM25 finds the exact error code, product
 * name or number that an embedding blurs. RRF needs no score calibration between the two:
 * it only uses each retriever's ranks.
 *
 * No re-ranker, on purpose for now: RRF over two retrievers is the cheap baseline, a
 * cross-encoder or LLM re-rank adds latency to every document answer, and the gold set's
 * recall@5 is the number that decides whether it has to earn its place (Module 3).
 */

export interface DocHit {
  chunkId: string;
  docId: string;
  title: string;
  text: string;
  locator: Locator;
  /** The fused score, for ordering. */
  rrf: number;
  /** Atlas vector score ((1 + cos) / 2) when the vector retriever found it. */
  vectorScore?: number;
}

export interface DocSearch {
  hits: DocHit[];
  /** The best vector score in the Space for this query: what mode=auto's router reads. */
  bestVectorScore: number;
  embedTokens: number;
}

type Ranked = { _id: string; score: number };

export async function searchDocuments(spaceId: string, query: string, signal?: AbortSignal, k = env.docsTopK): Promise<DocSearch> {
  const { vector, tokens } = await embed(query, signal);
  const [dense, lexical] =
    env.vectorBackend === 'mongo-cosine-scan'
      ? await scanBoth(spaceId, query, vector)
      : await Promise.all([vectorRanked(spaceId, vector), textRanked(spaceId, query)]);

  // Reciprocal rank fusion.
  const fused = new Map<string, { rrf: number; vectorScore?: number }>();
  dense.forEach((h, i) => fused.set(h._id, { rrf: 1 / (env.rrfK + i + 1), vectorScore: h.score }));
  lexical.forEach((h, i) => {
    const cur = fused.get(h._id) ?? { rrf: 0 };
    cur.rrf += 1 / (env.rrfK + i + 1);
    fused.set(h._id, cur);
  });
  const top = [...fused.entries()].sort((a, b) => b[1].rrf - a[1].rrf).slice(0, k);
  if (top.length === 0) return { hits: [], bestVectorScore: 0, embedTokens: tokens };

  // Hydrate: chunk text and locator, plus the document title a citation shows.
  const rows = await (await chunks())
    .find({ _id: { $in: top.map(([id]) => id) } }, { projection: { embedding: 0 } })
    .toArray();
  const byId = new Map(rows.map((r) => [r._id, r]));
  const docRows = await (await documents())
    .find({ _id: { $in: [...new Set(rows.map((r) => r.docId))] as DocumentDoc['_id'][] } }, { projection: { title: 1, status: 1 } })
    .toArray();
  const docById = new Map(docRows.map((d) => [d._id, d]));

  const hits: DocHit[] = [];
  for (const [id, s] of top) {
    const c = byId.get(id);
    const d = c && docById.get(c.docId);
    if (!c || !d || d.status === 'failed') continue; // a deleted or failed document is not citable
    hits.push({ chunkId: id, docId: c.docId, title: d.title, text: c.text, locator: c.locator, rrf: s.rrf, vectorScore: s.vectorScore });
  }
  return { hits, bestVectorScore: dense[0]?.score ?? 0, embedTokens: tokens };
}

async function vectorRanked(spaceId: string, queryVector: number[]): Promise<Ranked[]> {
  return (await chunks())
    .aggregate<Ranked>([
      {
        $vectorSearch: {
          index: SEARCH_INDEXES.chunksVector,
          path: 'embedding',
          queryVector,
          numCandidates: env.retrievalCandidates * 10,
          limit: env.retrievalCandidates,
          filter: { spaceId } // inside the stage: a $match afterwards would leak other Spaces into the top-k
        }
      },
      { $project: { _id: 1, score: { $meta: 'vectorSearchScore' } } }
    ])
    .toArray();
}

async function textRanked(spaceId: string, query: string): Promise<Ranked[]> {
  return (await chunks())
    .aggregate<Ranked>([
      {
        $search: {
          index: SEARCH_INDEXES.chunksText,
          compound: {
            must: [{ text: { query, path: 'text' } }],
            filter: [{ equals: { path: 'spaceId', value: spaceId } }]
          }
        }
      },
      { $limit: env.retrievalCandidates },
      { $project: { _id: 1, score: { $meta: 'searchScore' } } }
    ])
    .toArray();
}

/** Local mongod (no Atlas Search): cosine and a term-overlap score, both computed in Node. */
async function scanBoth(spaceId: string, query: string, vector: number[]): Promise<[Ranked[], Ranked[]]> {
  const all = await (await chunks()).find({ spaceId: spaceId as ChunkDoc['spaceId'] }, { projection: { text: 1, embedding: 1 } }).toArray();
  const terms = query.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 2);
  const dense = all
    .map((c) => ({ _id: c._id, score: (1 + cosine(vector, c.embedding)) / 2 }))
    .sort((a, b) => b.score - a.score)
    .slice(0, env.retrievalCandidates);
  const lexical = all
    .map((c) => {
      const text = c.text.toLowerCase();
      return { _id: c._id, score: terms.filter((t) => text.includes(t)).length };
    })
    .filter((c) => c.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, env.retrievalCandidates);
  return [dense, lexical];
}
