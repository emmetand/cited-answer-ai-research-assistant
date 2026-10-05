import { JSDOM, VirtualConsole } from 'jsdom';
import { Readability } from '@mozilla/readability';
import { isIP } from 'node:net';

export interface FetchedPage {
  url: string;
  title: string;
  /** The page's readable text, whitespace-collapsed. Citations' snippets are cut from this. */
  text: string;
}

const TIMEOUT_MS = 8_000;
const MAX_HTML_BYTES = 3_000_000;

/**
 * Download a page and extract its readable text. Throws with a specific message on any
 * failure (blocked, not HTML, empty), so the trace step can say exactly what went wrong.
 */
export async function fetchPage(rawUrl: string, signal: AbortSignal): Promise<FetchedPage> {
  const url = assertPublicHttpUrl(rawUrl);
  const res = await fetch(url, {
    redirect: 'follow',
    headers: {
      'user-agent': 'Mozilla/5.0 (compatible; CitedBot/0.1; +course project)',
      accept: 'text/html,application/xhtml+xml,text/plain;q=0.9'
    },
    signal: AbortSignal.any([signal, AbortSignal.timeout(TIMEOUT_MS)])
  });
  if (!res.ok) throw new Error(`${res.status} from ${url.hostname}`);

  const type = res.headers.get('content-type') ?? '';
  // Modern pages are mostly inline script and style; drop them before sizing and parsing.
  const body = (await res.text())
    .replace(/<script\b[\s\S]*?<\/script>/gi, '')
    .replace(/<style\b[\s\S]*?<\/style>/gi, '')
    .replace(/<svg\b[\s\S]*?<\/svg>/gi, '');
  if (body.length > MAX_HTML_BYTES) throw new Error(`page too large (${body.length} bytes of markup)`);

  if (type.includes('text/plain')) {
    return { url: res.url, title: url.hostname, text: collapse(body) };
  }
  if (!type.includes('html')) throw new Error(`unsupported content-type: ${type || 'none'}`);

  // A space before every tag, so adjacent elements never fuse into one word: without it
  // <li>Courses</li><li>Tutorials</li> reads as "CoursesTutorials", a word that exists
  // nowhere on the real page, and a citation quoting it fails the grounding check.
  // A silent virtual console: real pages are full of CSS jsdom cannot parse.
  const dom = new JSDOM(body.replace(/</g, ' <'), { url: res.url, virtualConsole: new VirtualConsole() });
  const article = new Readability(dom.window.document).parse();
  const text = collapse(article?.textContent || dom.window.document.body?.textContent || '');
  const title = collapse(article?.title || dom.window.document.title || url.hostname);
  dom.window.close();

  if (text.length < 200) throw new Error('no readable text on page (likely JavaScript-rendered)');
  return { url: res.url, title, text };
}

const collapse = (s: string) => s.replace(/\s+/g, ' ').trim();

/** The model chooses these URLs, so they are untrusted: public http(s) only. */
function assertPublicHttpUrl(raw: string): URL {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`not a valid URL: ${raw}`);
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new Error(`refusing ${url.protocol} URL`);
  const host = url.hostname.replace(/^\[|\]$/g, '');
  const privateHost =
    host === 'localhost' ||
    host.endsWith('.local') ||
    host.endsWith('.internal') ||
    (isIP(host) !== 0 && /^(10\.|127\.|169\.254\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|0\.|::1$|f[cd])/i.test(host));
  if (privateHost) throw new Error(`refusing private address ${host}`);
  return url;
}

/**
 * Is this running text, or a menu / banner / infobox? Navigation is mostly Capitalised
 * Words with no sentences; prose is mostly lowercase words that end in a full stop. A
 * menu also repeats the page title's words, which is why it otherwise wins on overlap.
 */
function isProse(text: string): boolean {
  const words = text.match(/[A-Za-z][A-Za-z'-]*/g) ?? [];
  if (words.length < 8) return false;
  const lower = words.filter((w) => w[0] === w[0]!.toLowerCase()).length;
  return lower / words.length >= 0.55 && /[a-z][.!?](\s|$)/.test(text);
}

/**
 * The parts of a page worth paying to send to the answer step: the opening (what the page
 * is), then the passages that share the most words with the question, in page order and
 * verbatim, up to `maxChars`. A whole page costs ~1 500 tokens; most of it is navigation,
 * boilerplate and sections about something else.
 */
export function focusedExcerpt(text: string, query: string, maxChars = 2000): string {
  if (text.length <= maxChars) return text;
  const terms = new Set(
    query
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length > 3)
  );
  // ~300-character windows on sentence boundaries.
  const sentences = text.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g) ?? [text];
  const windows: { i: number; text: string; score: number }[] = [];
  let cur = '';
  for (const s of sentences) {
    cur += s;
    if (cur.length >= 300) {
      windows.push({ i: windows.length, text: cur.trim(), score: 0 });
      cur = '';
    }
  }
  if (cur.trim()) windows.push({ i: windows.length, text: cur.trim(), score: 0 });
  for (const w of windows) {
    const words = new Set(w.text.toLowerCase().split(/[^a-z0-9]+/));
    w.score = isProse(w.text) ? [...terms].filter((t) => words.has(t)).length : 0;
  }

  // The opening says what the page is: the first stretch of prose, not the site's menu.
  const opening = windows.find((w) => isProse(w.text)) ?? windows[0]!;
  const chosen = new Set<number>([opening.i]);
  let used = opening.text.length;
  for (const w of [...windows].sort((a, b) => b.score - a.score || a.i - b.i)) {
    if (chosen.has(w.i) || w.score === 0) continue;
    if (used + w.text.length > maxChars) continue;
    chosen.add(w.i);
    used += w.text.length;
  }
  return windows
    .filter((w) => chosen.has(w.i))
    .map((w, k, arr) => (k > 0 && arr[k - 1]!.i !== w.i - 1 ? `… ${w.text}` : w.text))
    .join(' ');
}

/**
 * Pick the passage of a page that best matches the question, verbatim, to serve as the
 * citation's snippet. Verbatim matters: the grounding check looks for a run of ~12
 * consecutive words of the snippet in the real page.
 */
export function bestPassage(text: string, query: string, maxChars = 420): string {
  const terms = new Set(
    query
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length > 3)
  );
  const sentences = text.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g) ?? [text];

  let best = { score: -1, passage: '' };
  for (let i = 0; i < sentences.length; i++) {
    let passage = '';
    for (let j = i; j < sentences.length && (passage + sentences[j]).length <= maxChars; j++) passage += sentences[j];
    if (!passage) passage = sentences[i]!.slice(0, maxChars);
    passage = passage.trim();
    if (passage.split(' ').length < 15 || !isProse(passage)) continue;
    const words = new Set(passage.toLowerCase().split(/[^a-z0-9]+/));
    const score = [...terms].filter((t) => words.has(t)).length;
    if (score > best.score) best = { score, passage };
  }
  return best.passage || text.slice(0, maxChars).trim();
}
