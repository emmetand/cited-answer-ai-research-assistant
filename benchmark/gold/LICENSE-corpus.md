# Test corpus: licence and provenance

The four documents in `corpus/` (and their Markdown sources in `sources/`), the gold
question set `rag_gold.jsonl` and the page manifest `pages.json` were written by the
**FDE Agent Engineering Bootcamp staff** and are released under **Creative Commons
Attribution 4.0 International (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

They are used here, unmodified, as the retrieval test set for the benchmark.

## Why an authored corpus

1. **A deterministic gold set.** Every fact has one home, so a recall@5 number means what it
   says. With third-party documents a question is often answerable from two places, and
   "the retriever missed it" becomes an argument.
2. **Stable page numbers.** `build-corpus.mjs` renders the Markdown to PDF with a fixed
   paginator, so `p. 3` is `p. 3` on every machine, and `pages.json` records where each
   heading landed. That is what makes page-level citations checkable.
3. **No licence question.** Everything here may be redistributed with attribution.

## Contents

| File | Subject |
|---|---|
| `corpus/retrieval-basics.pdf` | BM25, dense retrieval, hybrid search, RRF, chunking, recall@k |
| `corpus/vector-search-on-mongodb.pdf` | Atlas Vector Search, filters, eventual consistency, M0 limits |
| `corpus/agent-loops-and-failure.md` | the loop, caps, termination, silent fallbacks, grounding, traces |
| `corpus/streaming-and-latency.md` | SSE, buffering, TTFT, percentiles, 202 and decoupling |

The PDFs are generated from the Markdown sources, not maintained separately:

    node benchmark/gold/build-corpus.mjs

After editing a source, re-run it and re-check any gold item whose `page` may have moved
(`pages.json` tells you), then `node benchmark/gold/validate-gold.mjs`.
