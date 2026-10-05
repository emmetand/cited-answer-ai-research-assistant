import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import pino from 'pino';
import { COLLECTIONS, type Depth, type RunDoc, type RunLog } from '@cited/contract';
import { env } from './env.js';
import { db } from './db.js';

/**
 * One run log per answer (the RunLog shape in the contract), written to two places:
 *   - runs/<requestId>.json on local disk, for reading trajectories and auditing budgets;
 *   - the `runs` collection, because a deployed instance's disk does not outlive a deploy.
 *     `node scripts/export-runs.mjs` copies those back into runs/.
 *
 * A run the user abandoned (closed the tab) goes to runs/aborted/ only: it is not a failure
 * of the agent, and mixed into runs/ it would read as one.
 */

const log = pino({ level: env.logLevel });
const ROOT = join(env.runsDir, '..');

export interface RunRecord {
  requestId: string;
  userId: string;
  threadId: string;
  answerId: string;
  query: string;
  depth: Depth;
  run: Omit<RunLog, 'depth'>;
  aborted: boolean;
  /** Kept on the Mongo copy only (not part of the file on disk): what /stats aggregates. */
  stats: { ttftMs: number | null; searchCached: boolean; webSearches: number };
}

export async function recordRun(r: RunRecord): Promise<void> {
  const file: RunLog & { requestId: string; query: string } = {
    requestId: r.requestId,
    query: r.query,
    ...r.run,
    depth: r.depth
  };
  const folder = r.aborted ? join(ROOT, 'runs', 'aborted') : env.runsDir;

  // Losing a run log loses the evidence for an answer: log loudly, but never turn an answer
  // the user already received into an error after the fact.
  await Promise.all([
    (async () => {
      await mkdir(folder, { recursive: true });
      await writeFile(join(folder, `${r.requestId.replace(/[^A-Za-z0-9_-]/g, '_')}.json`), JSON.stringify(file, null, 2));
    })().catch((err) => log.error({ requestId: r.requestId, err: (err as Error).message }, 'run log file write failed')),

    r.aborted
      ? Promise.resolve()
      : (async () => {
          const doc: RunDoc & RunRecord['stats'] = {
            ...r.run,
            ...r.stats,
            depth: r.depth,
            requestId: r.requestId,
            userId: r.userId,
            threadId: r.threadId as RunDoc['threadId'],
            answerId: r.answerId as RunDoc['answerId'],
            query: r.query,
            createdAt: new Date()
          };
          await (await db()).collection<RunDoc & RunRecord['stats']>(COLLECTIONS.runs).insertOne(doc);
        })().catch((err) => log.error({ requestId: r.requestId, err: (err as Error).message }, 'run log Mongo write failed'))
  ]);
}
