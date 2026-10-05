import type { Locator } from '@cited/contract';
import { env } from './env.js';

/**
 * Turn an upload into chunks, each with a locator a citation can point at:
 *   PDF       → { page }            (chunks never cross a page, so the page is exact)
 *   Markdown  → { heading, line }   (the section it sits in, and where that section starts)
 *   Text      → { line }
 *
 * `context` is prepended to the text only for the embedding: it tells the vector which
 * document and section a passage belongs to, without changing the text a citation quotes.
 */
export interface ParsedChunk {
  text: string;
  locator: Locator;
  context: string;
}

export interface Parsed {
  pages?: number;
  chunks: ParsedChunk[];
}

export async function parseUpload(bytes: Buffer, mimeType: string, title: string): Promise<Parsed> {
  if (mimeType === 'application/pdf') return parsePdf(bytes, title);
  const text = bytes.toString('utf8');
  if (mimeType === 'text/markdown') return { chunks: parseMarkdown(text, title) };
  return { chunks: parsePlain(text, title) };
}

async function parsePdf(bytes: Buffer, title: string): Promise<Parsed> {
  // The legacy build is the one that runs in Node.
  const { getDocument } = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const pdf = await getDocument({
    data: new Uint8Array(bytes),
    isEvalSupported: false,
    useSystemFonts: true,
    verbosity: 0 // errors only: real-world PDFs are full of harmless structural warnings
  }).promise;
  const out: ParsedChunk[] = [];
  try {
    for (let p = 1; p <= pdf.numPages; p++) {
      const page = await pdf.getPage(p);
      const content = await page.getTextContent();
      let text = '';
      for (const item of content.items) {
        if (!('str' in item)) continue;
        text += item.str + (item.hasEOL ? '\n' : ' ');
      }
      page.cleanup();
      for (const piece of splitText(text)) out.push({ text: piece, locator: { page: p }, context: `${title}, p. ${p}` });
    }
    return { pages: pdf.numPages, chunks: out };
  } finally {
    await pdf.destroy();
  }
}

function parseMarkdown(md: string, title: string): ParsedChunk[] {
  const lines = md.split(/\r?\n/);
  const sections: { heading?: string; line: number; body: string[] }[] = [{ line: 1, body: [] }];
  lines.forEach((raw, i) => {
    const h = raw.match(/^#{1,6}\s+(.*)$/);
    if (h) sections.push({ heading: h[1]!.trim(), line: i + 1, body: [] });
    else sections.at(-1)!.body.push(raw);
  });
  const out: ParsedChunk[] = [];
  for (const s of sections) {
    for (const piece of splitText(s.body.join('\n'))) {
      out.push({
        text: piece,
        locator: s.heading ? { heading: s.heading, line: s.line } : { line: s.line },
        context: s.heading ? `${title} — ${s.heading}` : title
      });
    }
  }
  return out;
}

function parsePlain(text: string, title: string): ParsedChunk[] {
  // Paragraphs keep their starting line, so a chunk's locator says where it begins.
  const lines = text.split(/\r?\n/);
  const paragraphs: { line: number; text: string }[] = [];
  let start = -1;
  let buf: string[] = [];
  lines.forEach((l, i) => {
    if (l.trim()) {
      if (start < 0) start = i + 1;
      buf.push(l);
    } else if (buf.length) {
      paragraphs.push({ line: start, text: buf.join(' ') });
      buf = [];
      start = -1;
    }
  });
  if (buf.length) paragraphs.push({ line: start, text: buf.join(' ') });

  const out: ParsedChunk[] = [];
  let cur: { line: number; text: string } | null = null;
  for (const p of paragraphs) {
    if (cur && cur.text.length + p.text.length > env.chunkChars) {
      for (const piece of splitText(cur.text)) out.push({ text: piece, locator: { line: cur.line }, context: title });
      cur = null;
    }
    cur = cur ? { line: cur.line, text: `${cur.text}\n\n${p.text}` } : { ...p };
  }
  if (cur) for (const piece of splitText(cur.text)) out.push({ text: piece, locator: { line: cur.line }, context: title });
  return out;
}

/**
 * Sentence-aware splitting into ~chunkChars pieces with ~chunkOverlapChars of overlap, so a
 * fact that straddles a boundary is whole in at least one chunk.
 */
export function splitText(raw: string, size = env.chunkChars, overlap = env.chunkOverlapChars): string[] {
  const text = raw.replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  if (!text) return [];
  if (text.length <= size) return [text];

  const sentences = text.match(/[^.!?\n]+(?:[.!?]+["')\]]*|\n+|$)\s*/g) ?? [text];
  const out: string[] = [];
  let cur: string[] = [];
  let len = 0;
  const flush = () => {
    const chunk = cur.join('').trim();
    if (chunk) out.push(chunk);
    // carry the last sentences forward as overlap
    const keep: string[] = [];
    let kept = 0;
    for (let i = cur.length - 1; i >= 0 && kept + cur[i]!.length <= overlap; i--) {
      keep.unshift(cur[i]!);
      kept += cur[i]!.length;
    }
    cur = keep;
    len = kept;
  };
  for (let s of sentences) {
    while (s.length > size) {
      // one enormous "sentence" (a table, a code block): hard-split it
      if (len) flush();
      out.push(s.slice(0, size).trim());
      s = s.slice(size - overlap);
    }
    if (len + s.length > size && len > overlap) flush();
    cur.push(s);
    len += s.length;
  }
  if (cur.join('').trim() && len > 0) {
    const last = cur.join('').trim();
    if (!out.length || !out.at(-1)!.endsWith(last)) out.push(last);
  }
  return out.filter((c) => c.length >= 20);
}
