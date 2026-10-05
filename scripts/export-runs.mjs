#!/usr/bin/env node
/**
 * Dump the `runs` collection into runs/<requestId>.json.
 *
 *   node scripts/export-runs.mjs                     # from MONGODB_URI in .env
 *   node scripts/export-runs.mjs --limit 200
 *
 * Locally, the agent writes a run log per answer to disk. A deployed instance's disk does
 * not outlive a deploy, so it also writes them to Mongo; this brings them back to disk for
 * reading trajectories and auditing budgets.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { config } from 'dotenv';
import { MongoClient } from 'mongodb';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
config({ path: join(ROOT, '.env') });

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i + 1] : fallback;
};

const uri = process.env.MONGODB_URI;
if (!uri) {
  console.error('MONGODB_URI is not set.');
  process.exit(2);
}

const limit = Number(arg('limit', 500));
const outDir = resolve(ROOT, arg('out', 'runs'));
mkdirSync(outDir, { recursive: true });

const client = new MongoClient(uri, { serverSelectionTimeoutMS: 8000 });
try {
  await client.connect();
  const runs = await client
    .db(process.env.MONGODB_DB ?? 'cited')
    .collection('runs')
    .find({}, { sort: { createdAt: -1 }, limit })
    .toArray();

  if (!runs.length) {
    console.log('no runs in Mongo yet — has your ask path written any?');
    process.exit(0);
  }

  for (const run of runs) {
    const id = run.requestId ?? String(run._id);
    // Only the RunLog fields, so the file on disk is the declared shape.
    const { tokens, wallClockSec, costUsd, terminated, toolCalls } = run;
    writeFileSync(
      join(outDir, `${id}.json`),
      JSON.stringify({ tokens, wallClockSec, costUsd, terminated, toolCalls }, null, 2)
    );
  }
  console.log(`wrote ${runs.length} run log(s) to ${outDir}`);
} finally {
  await client.close();
}
