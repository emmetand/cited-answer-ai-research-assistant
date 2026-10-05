/**
 * The jobs worker: its own process (`npm run worker`), so parsing a 60-page PDF never
 * competes with the process that is streaming somebody's answer.
 *
 *   loop: sweep stale claims → atomically claim the oldest pending job → run it → record the outcome
 *
 * Crash safety: a worker killed mid-job leaves its row `running` with an old `claimedAt`.
 * The sweeper (any worker, on its next pass) returns it to `pending`, and the re-run skips
 * the stages that already finished (see ingest.ts).
 *
 * Deep search does NOT run here: it streams over the same SSE channel as a quick answer,
 * because someone watching a deep search wants to see it work, not poll a job id.
 */
import { hostname } from 'node:os';
import { setTimeout as sleep } from 'node:timers/promises';
import pino from 'pino';
import type { DocumentDoc } from '@lumina/contract';
import { env } from './env.js';
import { documents, jobs } from './documents.js';
import { indexDocument } from './ingest.js';

const log = pino({ level: env.logLevel }).child({ worker: `wkr_${hostname()}_${process.pid}` });
const workerId = `wkr_${hostname()}_${process.pid}`;

const POLL_MS = 1000;
const SWEEP_EVERY_MS = 30_000;
/** A claim older than this is a crashed worker, not a slow one (the probe alone may take 90 s). */
const STALE_AFTER_MS = 10 * 60_000;
const MAX_ATTEMPTS = 3;

let stopping = false;
for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    stopping = true; // finish the current job, then exit
    log.info({ sig }, 'stopping after the current job');
  });
}

async function sweep() {
  const res = await (await jobs()).updateMany(
    { status: 'running', claimedAt: { $lt: new Date(Date.now() - STALE_AFTER_MS) } },
    { $set: { status: 'pending' }, $unset: { claimedAt: '', workerId: '' } }
  );
  if (res.modifiedCount) log.warn({ jobs: res.modifiedCount }, 'swept stale claims back to pending');
}

async function claim() {
  // Atomic: two workers polling at once can never both get the same row.
  return (await jobs()).findOneAndUpdate(
    { status: 'pending' },
    { $set: { status: 'running', claimedAt: new Date(), workerId }, $inc: { attempts: 1 } },
    { sort: { createdAt: 1 }, returnDocument: 'after' }
  );
}

async function run(job: NonNullable<Awaited<ReturnType<typeof claim>>>) {
  const docId = String(job.payload.docId ?? '');
  const t0 = Date.now();
  log.info({ job: job._id, kind: job.kind, docId, attempt: job.attempts }, 'job claimed');
  try {
    if (job.kind !== 'index_document') throw new Error(`unknown job kind ${job.kind}`);
    await indexDocument(docId, (msg, extra) => log.info({ job: job._id, ...extra }, msg));
    await (await jobs()).updateOne({ _id: job._id }, { $set: { status: 'done' }, $unset: { error: '' } });
    log.info({ job: job._id, docId, ms: Date.now() - t0 }, 'job done');
  } catch (err) {
    const error = (err as Error).message || 'unknown error';
    const retry = job.attempts < MAX_ATTEMPTS;
    await (await jobs()).updateOne(
      { _id: job._id },
      { $set: { status: retry ? 'pending' : 'failed', error }, $unset: { claimedAt: '' } }
    );
    // The document says what happened: failed for good, or back in the queue for another try.
    await (await documents()).updateOne(
      { _id: docId as DocumentDoc['_id'] },
      retry ? { $set: { status: 'pending', error: `attempt ${job.attempts} failed, retrying: ${error}` } } : { $set: { status: 'failed', error } }
    );
    log.error({ job: job._id, docId, attempt: job.attempts, retry, err: error }, 'job failed');
  }
}

async function main() {
  log.info({ pollMs: POLL_MS }, 'jobs worker up');
  let lastSweep = 0;
  while (!stopping) {
    try {
      if (Date.now() - lastSweep > SWEEP_EVERY_MS) {
        await sweep();
        lastSweep = Date.now();
      }
      const job = await claim();
      if (job) await run(job);
      else await sleep(POLL_MS);
    } catch (err) {
      // Mongo unreachable, say: back off and keep the worker alive.
      log.error({ err: (err as Error).message }, 'worker loop error');
      await sleep(5000);
    }
  }
  process.exit(0);
}

void main();
