/**
 * @cited/contract — the API contract, as executable schemas.
 *
 * The gateway validates inbound requests with these schemas, the agent service validates
 * the events it emits, and the React UI compiles against the inferred types. One
 * definition, three consumers: drift fails `npm run typecheck` before it fails a user.
 *
 * Every route, status code, SSE event and MongoDB document is defined here.
 */
export * from './ids.js';
export * from './sse.js';
export * from './http.js';
export * from './db.js';
