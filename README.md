# markitdownwebsearcher

A local-first Model Context Protocol (MCP) server that gives LLM clients (Claude Desktop, Cursor, Windsurf) a low-token web search tool. Instead of dumping whole pages into context, it retrieves results across many domains, reranks them locally with two small transformer models, deduplicates overlapping passages, and returns only the most relevant excerpts, sorted newest-first.
This is a single-user, local-use tool. It is not designed or audited for multi-tenant or hostile network deployment.

## What it does
Given a query, the server:

1. Pulls result links from a local SearXNG instance (paginated across multiple result pages, accumulating distinct domains), with a DuckDuckGo HTML fallback if SearXNG returns nothing.
2. Fetches each result page through an SSRF-hardened, IP-pinned fetcher with a streaming byte ceiling.
3. Extracts article text, segments it into overlapping sentence windows, and filters out keyword-spam blocks.
4. Reranks passages locally: a bi-encoder pre-filter followed by a cross-encoder rerank with sigmoid-normalized scoring.
5. Deduplicates near-identical passages and returns excerpts (capped per domain), sorted newest-first by publication date where available.

It returns text excerpts with their source URLs and dates — not a synthesized answer. Your LLM client does the synthesis.


## Features

- **Local two-stage reranking** — a bi-encoder (all-MiniLM-L6-v2) pre-filter followed by a cross-encoder (ms-marco-MiniLM-L-6-v2) rerank. No external ranking API and no per-query cost.
- **SSRF-validated fetch** — each hostname is resolved once and the resolved IP is checked against private/loopback/link-local/reserved/multicast/unspecified/CGNAT ranges before connecting. The connection is pinned to the validated IP, with TLS SNI and certificate validation against the real hostname. Redirects are followed manually with a hop limit, re-validating at each hop.
- **Streaming byte ceiling** — downloads abort once they exceed the configured limit (currently 5 MB; see MAX_PAGE_SIZE_BYTES) to keep memory bounded under concurrent fetches.
- **Deduplication** — Jaccard-similarity filtering on longer overlapping passages.
- **Token report** — reports the returned excerpt token count against the raw extracted token count (a rough local heuristic, not a benchmark against any other tool).
- **Dual-pass recency** — one time-filtered pass (recency-biased) plus one unfiltered pass to backfill evergreen sources.

## Requirements

- Python 3.10+.
- A reachable SearXNG instance with JSON output enabled (default expected at http://localhost:8080/search). Without it, the tool falls back to scraping DuckDuckGo's HTML endpoint, which is best-effort only (see Limitations).
-   Install dependencies:

```bash
pip install -r requirements.txt
```

First run downloads the two models (~180 MB) from Hugging Face and caches them.

## Use with Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config),
using absolute paths:

```json
{
  "mcpServers": {
    "deep-search": {
      "command": "C:\\path\\to\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\markitdownwebsearcher.py"]
    }
  }
}
```

Fully quit and reopen Claude Desktop. The `deep_search` tool should appear.

## Configuration

Key constants at the top of `markitdownwebsearcher.py`:

| Constant | Meaning |
| --- | --- |
| `SEARXNG_URL` | Local SearXNG search endpoint |
| `MAX_PAGE_SIZE_BYTES` | Per-page download ceiling (streaming abort) |
| `TARGET_DISTINCT_DOMAINS` | Domain-count goal that stops pagination |
| `MAX_PAGES_TO_ACCUMULATE` | Hard ceiling on pagination depth |
| `SEARXNG_TIME_RANGE` | Recency window for the time-filtered pass |
| `DUAL_PASS_RECENCY` | Whether to run the second, unfiltered pass |
| `MAX_CANDIDATES_POOL` | Passages kept after the bi-encoder pre-filter |
| `MAX_RESULTS` | Maximum excerpts returned |
| `MAX_PER_DOMAIN` | Maximum excerpts from any single domain |
| `MAX_FETCH_WORKERS` | Concurrent fetch workers |

If you change any of these, update the comments next to them so the values and the prose stay in sync.

## Limitations & scope

- **Single-user, local use.** The fetch path is SSRF-validated but not audited
  for hostile multi-tenant deployment.
- **Search depends on scraping the DuckDuckGo HTML endpoint** (`html.duckduckgo.com`).
  This is unofficial, may break without notice, and is subject to DuckDuckGo's
  terms of service. Treat search reliability as best-effort.
- Pages are fetched sequentially with no concurrency; large source counts
  increase latency, which can hit a client's tool-call timeout.
- Large pages (e.g. some Wikipedia articles) may be skipped by the 512 KB ceiling.

## License

[MIT ]
