import type { Response } from 'express';
import {
  DoneEvent,
  PlanEvent,
  SourcesEvent,
  StreamErrorEvent,
  TokenEvent,
  TraceEvent,
  type SseEventName
} from '@cited/contract';

const schemas = {
  plan: PlanEvent,
  trace: TraceEvent,
  sources: SourcesEvent,
  token: TokenEvent,
  done: DoneEvent,
  error: StreamErrorEvent
} as const;

/**
 * One SSE stream for one answer. Headers go out lazily on the first event, so a provider
 * that fails before anything has streamed still gets a real 502 status instead of a 200
 * stream carrying an error event.
 *
 * Every event is validated against the contract before it is written: a malformed event
 * is our bug, and it should fail here rather than in the UI.
 */
export class SseStream {
  private opened = false;
  private ended = false;

  constructor(private readonly res: Response) {}

  get isOpen() {
    return this.opened;
  }

  private open() {
    if (this.opened) return;
    this.opened = true;
    this.res.status(200);
    this.res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    this.res.setHeader('Cache-Control', 'no-cache, no-transform');
    this.res.setHeader('Connection', 'keep-alive');
    // Proxies (nginx, Fly's edge) buffer unless told not to; buffered SSE is batch in a costume.
    this.res.setHeader('X-Accel-Buffering', 'no');
    this.res.flushHeaders();
    this.res.socket?.setNoDelay(true);
  }

  send(event: SseEventName, data: unknown) {
    if (this.ended || this.res.writableEnded) return;
    const parsed = schemas[event].parse(data);
    this.open();
    this.res.write(`event: ${event}\ndata: ${JSON.stringify(parsed)}\n\n`);
  }

  end() {
    if (this.ended) return;
    this.ended = true;
    if (!this.res.writableEnded) this.res.end();
  }
}
