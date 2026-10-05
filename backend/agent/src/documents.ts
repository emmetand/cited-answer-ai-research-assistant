import { randomUUID } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { GridFSBucket, ObjectId } from 'mongodb';
import {
  COLLECTIONS,
  GRIDFS_BUCKETS,
  newId,
  type ChunkDoc,
  type DocumentDoc,
  type JobDoc,
  type SpaceDoc
} from '@lumina/contract';
import { db } from './db.js';

/** Spaces, documents, raw uploads (GridFS) and the jobs queue: everything on disk for RAG. */

export const spaces = async () => (await db()).collection<SpaceDoc>(COLLECTIONS.spaces);
export const documents = async () => (await db()).collection<DocumentDoc>(COLLECTIONS.documents);
export const chunks = async () => (await db()).collection<ChunkDoc>(COLLECTIONS.chunks);
export const jobs = async () => (await db()).collection<JobDoc>(COLLECTIONS.jobs);
const uploads = async () => new GridFSBucket(await db(), { bucketName: GRIDFS_BUCKETS.uploads });

export async function createSpace(userId: string, name: string) {
  const space: SpaceDoc = { _id: newId('spc') as SpaceDoc['_id'], userId, name, createdAt: new Date() };
  await (await spaces()).insertOne(space);
  return space;
}

export async function findSpace(userId: string, spaceId: string) {
  return (await spaces()).findOne({ _id: spaceId as SpaceDoc['_id'], userId });
}

/** The extension wins over a browser's guess: .md often arrives as application/octet-stream. */
export function mimeFor(filename: string, reported: string): string | null {
  const ext = filename.toLowerCase().split('.').pop();
  if (ext === 'pdf') return 'application/pdf';
  if (ext === 'md' || ext === 'markdown') return 'text/markdown';
  if (ext === 'txt') return 'text/plain';
  if (['application/pdf', 'text/markdown', 'text/plain'].includes(reported)) return reported;
  return null;
}

/** Put the raw bytes in GridFS. Returns the GridFS id. */
export async function storeFile(filename: string, bytes: Buffer, metadata: Record<string, unknown>): Promise<string> {
  const upload = (await uploads()).openUploadStream(filename, { metadata });
  await pipeline(Readable.from([bytes]), upload);
  return upload.id.toString();
}

export async function deleteFile(fileId: string) {
  await (await uploads()).delete(new ObjectId(fileId));
}

/**
 * Everything the 202 promises, and nothing more: the bytes are already safe in GridFS
 * (storeFile), and now the document row says `pending` and a job row is waiting. Parsing,
 * chunking and embedding happen on the worker. The two inserts are independent, so they
 * run in parallel: the accept has a 300 ms budget and every round trip to Atlas counts.
 */
export async function registerUpload(
  docId: DocumentDoc['_id'],
  fileId: string,
  userId: string,
  spaceId: string,
  filename: string,
  mimeType: string,
  bytes: number
) {
  const now = new Date();
  await Promise.all([
    (await documents()).insertOne({
      _id: docId,
      spaceId: spaceId as DocumentDoc['spaceId'],
      userId,
      title: filename,
      mimeType,
      bytes,
      status: 'pending',
      pct: 0,
      fileId,
      createdAt: now
    }),
    (await jobs()).insertOne({
      _id: `job_${randomUUID()}`,
      kind: 'index_document',
      status: 'pending',
      payload: { docId },
      userId,
      attempts: 0,
      createdAt: now
    })
  ]);
}

export const newDocId = () => newId('doc') as DocumentDoc['_id'];

export async function readUpload(fileId: string): Promise<Buffer> {
  const parts: Buffer[] = [];
  for await (const part of (await uploads()).openDownloadStream(new ObjectId(fileId))) parts.push(part as Buffer);
  return Buffer.concat(parts);
}

export async function listDocuments(spaceId: string) {
  const rows = await (await documents()).find({ spaceId: spaceId as DocumentDoc['spaceId'] }).sort({ createdAt: -1 }).toArray();
  return rows.map((d) => ({
    docId: d._id,
    title: d.title,
    status: d.status,
    pct: d.pct,
    ...(d.pages ? { pages: d.pages } : {}),
    ...(d.chunks !== undefined ? { chunks: d.chunks } : {}),
    ...(d.error ? { error: d.error } : {})
  }));
}
