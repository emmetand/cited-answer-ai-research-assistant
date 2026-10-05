import { MongoServerError } from 'mongodb';
import { env } from './env.js';
import { db } from './db.js';

/**
 * The deep-search spend gate: DEEP_DAILY_CAP deep searches per X-User-Id per UTC day,
 * enforced here in the agent service, next to the spending. (On the gateway it would be a
 * cap anyone could bypass by calling the agent directly.)
 *
 * One counter document per user per day. Taking a slot is a single atomic update that only
 * matches while the count is under the cap, so two simultaneous requests cannot both take
 * the last slot.
 */

interface UsageDoc {
  _id: string; // `${userId}:${YYYY-MM-DD}`
  userId: string;
  day: string;
  count: number;
  updatedAt: Date;
}

const usage = async () => (await db()).collection<UsageDoc>('deepUsage');
const today = () => new Date().toISOString().slice(0, 10);

/** When today's allowance resets: the next UTC midnight. */
export function resetsAt(): string {
  const d = new Date();
  return new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1)).toISOString();
}

/** Take one deep search from today's allowance. False when the cap is already reached. */
export async function takeDeepSlot(userId: string): Promise<boolean> {
  // A cap of 0 means deep is switched off. (Without this, the upsert below would create
  // the day's first counter at 1 and let one request through.)
  if (env.deepDailyCap <= 0) return false;
  const day = today();
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await (await usage()).findOneAndUpdate(
        { _id: `${userId}:${day}`, count: { $lt: env.deepDailyCap } },
        { $inc: { count: 1 }, $set: { userId, day, updatedAt: new Date() } },
        { upsert: true, returnDocument: 'after' }
      );
      return res !== null;
    } catch (err) {
      // Duplicate key means the filter missed an existing document, for one of two reasons:
      //   - it is at the cap (the upsert then tried to insert a second copy): over cap;
      //   - two requests both tried to CREATE today's counter, and this one lost the race.
      // Retrying once tells them apart: the counter now exists, so the update path decides.
      if (err instanceof MongoServerError && err.code === 11000) {
        if (attempt < 2) continue;
        return false;
      }
      throw err;
    }
  }
}

export async function deepUsedToday(userId: string): Promise<number> {
  const doc = await (await usage()).findOne({ _id: `${userId}:${today()}` });
  return doc?.count ?? 0;
}
