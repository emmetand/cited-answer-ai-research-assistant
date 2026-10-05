import { setTimeout as sleep } from 'node:timers/promises';
import type { AnyBulkWriteOperation } from 'mongodb';
import { SEARCH_INDEXES, type ChunkDoc, type DocumentDoc } from '@lumina/contract';
import { env } from './env.js';
import { chunks, documents, readUpload } from './documents.js';
import { parseUpload } from './parse.js';
import { embedMany } from './embeddings.js';

/**
 * index_document: GridFS → parse → chunk → embed → upsert → read-your-write probe → indexed.
 *
 * Status only moves forward after the work behind it succeeded, and `indexed` is earned by
 * the probe, not by the upsert. Re-running a job after a crash re-parses (cheap) but only
 * embeds chunks that are not already stored (the expensive, paid stage is not repeated).
 */

type Log = (msg: string, extra?: Record<string, unknown>) => void;

export async function indexDocument(docId: string, log: Log): Promise<void> {
  const docs = await documents();
  const doc = await docs.findOne({ _id: docId as DocumentDoc['_id'] });
  if (!doc) throw new Error(`document ${docId} no longer exists`);
  const setStatus = (fields: Partial<DocumentDoc>) => docs.updateOne({ _id: doc._id }, { $set: fields, $unset: { error: '' } });

  // ---- parse + chunk
  await setStatus({ status: 'parsing', pct: 5 });
  const bytes = await readUpload(doc.fileId);
  const parsed = await parseUpload(bytes, doc.mimeType, doc.title);
  if (parsed.chunks.length === 0) throw new Error('no extractable text (a scanned PDF needs OCR, which LUMINA does not do)');
  log('parsed', { docId, pages: parsed.pages, chunks: parsed.chunks.length });
  await setStatus({ status: 'embedding', pct: 15, ...(parsed.pages ? { pages: parsed.pages } : {}) });

  // ---- embed only what is not already stored (resume after a crash)
  const col = await chunks();
  const stored = new Set((await col.find({ docId: doc._id }, { projection: { ord: 1 } }).toArray()).map((c) => c.ord));
  const todo = parsed.chunks.map((c, ord) => ({ ...c, ord })).filter((c) => !stored.has(c.ord));
  const BATCH = 64;
  for (let i = 0; i < todo.length; i += BATCH) {
    const batch = todo.slice(i, i + BATCH);
    const { vectors } = await embedMany(batch.map((c) => `${c.context}\n\n${c.text}`));
    const ops: AnyBulkWriteOperation<ChunkDoc>[] = batch.map((c, j) => ({
      replaceOne: {
        filter: { _id: chunkId(doc._id, c.ord) },
        replacement: {
          docId: doc._id,
          spaceId: doc.spaceId,
          userId: doc.userId,
          text: c.text,
          locator: c.locator,
          ord: c.ord,
          embedding: vectors[j]!,
          createdAt: new Date()
        },
        upsert: true
      }
    }));
    await col.bulkWrite(ops, { ordered: false });
    const done = Math.min(i + BATCH, todo.length);
    await setStatus({ status: 'embedding', pct: 15 + Math.round((70 * done) / todo.length) });
  }
  // A re-parse that produced fewer chunks than a previous attempt: drop the leftovers.
  await col.deleteMany({ docId: doc._id, ord: { $gte: parsed.chunks.length } });
  log('embedded', { docId, embedded: todo.length, reused: stored.size });

  // ---- read-your-write probe: searchable, not just stored
  await setStatus({ status: 'embedding', pct: 90 });
  await probe(doc._id, doc.spaceId, log);

  await docs.updateOne(
    { _id: doc._id },
    { $set: { status: 'indexed', pct: 100, chunks: parsed.chunks.length }, $unset: { error: '' } }
  );
}

export const chunkId = (docId: string, ord: number) => `${docId}_c${ord}`;

/**
 * Atlas Search indexes are eventually consistent: a chunk can be in the collection for
 * several seconds before either search index returns it. Ask both indexes for a chunk we
 * just wrote, by its own vector and by its own words, inside its Space, until they answer.
 */
async function probe(docId: string, spaceId: string, log: Log) {
  const col = await chunks();
  const sample = await col.findOne({ docId: docId as ChunkDoc['docId'] }, { sort: { ord: 1 } });
  if (!sample) throw new Error('probe: no chunks were stored');
  if (env.vectorBackend === 'mongo-cosine-scan') return; // a plain collection scan is read-your-write already

  const words = sample.text.split(/\s+/).slice(0, 12).join(' ');
  const deadline = Date.now() + env.probeTimeoutSec * 1000;
  let vectorOk = false;
  let textOk = false;
  for (let attempt = 1; ; attempt++) {
    if (!vectorOk) {
      const hits = await col
        .aggregate<{ _id: string }>([
          {
            $vectorSearch: {
              index: SEARCH_INDEXES.chunksVector,
              path: 'embedding',
              queryVector: sample.embedding,
              numCandidates: 50,
              limit: 10,
              filter: { spaceId }
            }
          },
          { $project: { _id: 1 } }
        ])
        .toArray();
      vectorOk = hits.some((h) => h._id === sample._id);
    }
    if (!textOk) {
      const hits = await col
        .aggregate<{ _id: string }>([
          {
            $search: {
              index: SEARCH_INDEXES.chunksText,
              compound: {
                must: [{ text: { query: words, path: 'text' } }],
                filter: [{ equals: { path: 'spaceId', value: spaceId } }]
              }
            }
          },
          { $limit: 20 },
          { $project: { _id: 1 } }
        ])
        .toArray();
      textOk = hits.some((h) => h._id === sample._id);
    }
    if (vectorOk && textOk) {
      log('probe ok', { docId, attempt });
      return;
    }
    if (Date.now() > deadline) {
      throw new Error(
        `probe timed out after ${env.probeTimeoutSec}s: chunks are stored but not yet searchable (vector ${vectorOk ? 'ok' : 'missing'}, text ${textOk ? 'ok' : 'missing'})`
      );
    }
    await sleep(Math.min(1000 * attempt, 5000));
  }
}
