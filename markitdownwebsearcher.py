#!/usr/bin/env python3
"""
Deep Search MCP Server (Version 16.0)
Local-first JSON-RPC stdio search tool with IP-pinned SSRF-hardened fetch,
streaming byte ceiling, cross-encoder reranking, and honest token accounting.

v16 changes (all in the RETRIEVAL / RANKING / OUTPUT strategy; the fetcher,
SSRF hardening, redirect logic, and token accounting are unchanged):

  A. Paginated index accumulator (SearXNG) with three termination guards
     (target distinct domains / max pages / stall-break) and conservative,
     pre-fetch URL normalization that dedups before spinning up fetch workers.
  B. Sliding-window segmentation shifted to step=1 (2-sentence overlap) to keep
     figures glued to their qualifying clauses. No `len-2` tail truncation.
  C. Hybrid-Coverage reranker: cross-encoder demoted from gatekeeper to
     high-signal noise filter. Relative floor (top_prob - 0.60) PLUS a small
     absolute floor so the junk filter still bites on hard queries. `break`
     only on the floor; `continue` on domain-cap and dedup. Pool right-sized
     to 120 to balance the 3x segment growth from step=1.
  D. Dual-source chronological sort: engine-provided date (preferred) with a
     trafilatura bare_extraction scraped-date fallback, bound to each segment,
     sorted newest-first with undated entries trailing gracefully.

Scope note: intended for single-user, local-first desktop use (Claude Desktop,
Cursor, Windsurf). The fetch path validates resolved IPs and pins the connection
to the validated IP with correct SNI/cert validation; it is not audited for
hostile multi-tenant deployment.
"""

import sys
import json
import re
import socket
import math
import ipaddress
import datetime as dt
import tiktoken
from urllib.parse import (quote_plus, urlparse, parse_qs, urljoin,
                          urlencode, urlunparse)
from bs4 import BeautifulSoup
import trafilatura
from urllib3.poolmanager import PoolManager
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================================================================
# GLOBAL CONFIGURATION
# ======================================================================
MAX_PAGE_SIZE_BYTES = 2 * 1024 * 1024    # 2 MB ceiling: captures any real article,
                                         # keeps memory predictable under 8 workers.
MAX_REDIRECT_HOPS = 3
FETCH_TIMEOUT = 5.0
TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

# --- (A) Paginated index accumulator ---------------------------------
SEARXNG_URL = "http://localhost:8080/search"   # local trusted backend
TARGET_DISTINCT_DOMAINS = 50     # the "top 50 latest websites" goal
MAX_PAGES_TO_ACCUMULATE = 10     # hard ceiling on pagination depth
SEARXNG_CATEGORIES = "general,news"   # news engines populate dates + time_range
SEARXNG_TIME_RANGE = "month"     # recency bias; "" disables (see dual-pass note)
DUAL_PASS_RECENCY = True         # run one time-filtered + one unfiltered pass

# --- (C) Reranker (Hybrid-Coverage Mode) -----------------------------
MAX_CANDIDATES_POOL = 120        # right-sized for step=1's ~3x segment growth
MAX_RESULTS = 50                 # final excerpts returned (one per domain target)
RELATIVE_CUTOFF_MARGIN = 0.60    # keep chunks within 0.60 of the top probability
ABSOLUTE_PROB_FLOOR = 0.05       # hard junk floor; bites even when query matches poorly
JACCARD_DEDUP_THRESHOLD = 0.35
DEDUP_MIN_LEN = 180              # only dedup long narrative blocks
MAX_PER_DOMAIN = 1               # one excerpt per site -> 50 distinct websites

# --- Fetch concurrency ------------------------------------------------
MAX_FETCH_WORKERS = 8

# Tracking params stripped during normalization. Whitelist (remove ONLY these)
# rather than blanket-stripping the query string, since some sites encode the
# article id in a query param (?id=, ?p=, ?story=).
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_reader", "utm_name", "utm_social", "utm_brand",
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid",
    "igshid", "ref", "ref_src", "ref_url", "_hsenc", "_hsmi",
    "yclid", "twclid", "wt_mc", "spm",
}

# Initialize models globally (blocks the stdio thread during inference;
# acceptable for single-user desktop use).
try:
    print("Loading ML models...", file=sys.stderr)
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    sys.stderr.write(f"FATAL: Model init failed: {e}\n")
    sys.exit(1)

# ======================================================================
# SECURITY & FETCH ENGINE  (UNCHANGED from v15)
# ======================================================================

def resolve_and_verify_ip(hostname: str) -> str:
    """
    Resolve a hostname once and return the first IP that is NOT in any
    private/loopback/link-local/reserved/multicast/unspecified/CGNAT range.
    Returns None on block or DNS failure (with a stderr reason).
    """
    if not hostname:
        return None
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except Exception as ex:
        print(f"DNS RESOLUTION ERROR for '{hostname}': {ex}", file=sys.stderr)
        return None

    for item in addr_info:
        ip = item[4][0]
        if "%" in ip:
            ip = ip.split("%")[0]
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or
                ip_obj.is_reserved or ip_obj.is_multicast or ip_obj.is_unspecified or
                ip.startswith("100.64.")):
            print(f"SECURITY BLOCK: SSRF-disallowed IP {ip} for host '{hostname}'",
                  file=sys.stderr)
            return None
        return ip
    return None


def execute_hardened_fetch(target_url: str) -> str:
    """
    Fetch a URL with:
      - single DNS resolution + IP validation per hop (no TOCTOU within a hop),
      - connection pinned to the validated IP (IP in URL netloc),
      - correct SNI + cert validation against the real hostname
        (assert_hostname / server_hostname),
      - manual redirect loop (allow_redirects disabled),
      - streaming download aborted at MAX_PAGE_SIZE_BYTES (true byte ceiling).
    Returns decoded text, or None on any block/error (reason logged to stderr).
    """
    current_url = target_url
    for hop in range(MAX_REDIRECT_HOPS):
        try:
            parsed = urlparse(current_url)
            if parsed.scheme not in ("http", "https"):
                print(f"FETCH BLOCK: unsupported scheme '{parsed.scheme}'", file=sys.stderr)
                return None
            host = parsed.hostname
            if not host:
                return None

            ip = resolve_and_verify_ip(host)
            if not ip:
                return None

            # Pin the connection to the validated IP.
            ip_netloc = f"[{ip}]" if ":" in ip else ip
            if parsed.port:
                ip_netloc += f":{parsed.port}"
            rewritten_url = parsed._replace(netloc=ip_netloc).geturl()

            # TLS kwargs only apply to HTTPS; passing them to a plain HTTP
            # connection raises a TypeError in urllib3.
            pm_kwargs = {}
            if parsed.scheme == "https":
                pm_kwargs = {"assert_hostname": host, "server_hostname": host}
            pool = PoolManager(**pm_kwargs)

            headers = {
                "User-Agent": "Mozilla/5.0 (ContextDistiller/16.0)",
                "Host": host,
            }

            resp = pool.request(
                "GET", rewritten_url,
                headers=headers,
                redirect=False,
                preload_content=False,
                timeout=FETCH_TIMEOUT,
            )

            try:
                if resp.status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        return None
                    # Resolve the redirect against the real hostname, not the IP.
                    current_url = urljoin(current_url, location)
                    continue

                if resp.status != 200:
                    return None

                # Streaming byte ceiling: abort before exceeding the limit.
                buf = bytearray()
                total = 0
                for chunk in resp.stream(16384, decode_content=True):
                    total += len(chunk)
                    if total > MAX_PAGE_SIZE_BYTES:
                        print(f"STREAM OVERFLOW: aborted '{current_url}' "
                              f"(> {MAX_PAGE_SIZE_BYTES} bytes)", file=sys.stderr)
                        return None
                    buf.extend(chunk)
                return buf.decode("utf-8", errors="ignore")
            finally:
                resp.release_conn()

        except Exception as err:
            print(f"FETCH EXCEPTION on hop {hop} for '{current_url}': "
                  f"{type(err).__name__}: {err}", file=sys.stderr)
            return None

    print(f"REDIRECT LIMIT: exceeded {MAX_REDIRECT_HOPS} hops for '{target_url}'",
          file=sys.stderr)
    return None

# ======================================================================
# (A) PAGINATED INDEX ACCUMULATOR + URL NORMALIZATION
# ======================================================================

def normalize_url(url: str) -> str:
    """
    Conservative pre-fetch normalization for dedup:
      - lowercase scheme + host
      - drop fragment
      - drop trailing slash on the path
      - remove ONLY whitelisted tracking params (keep id-bearing params)
    Distinct articles that differ only by id-bearing query params are preserved.
    """
    try:
        p = urlparse(url)
    except Exception:
        return url
    if p.scheme not in ("http", "https"):
        return url

    scheme = p.scheme.lower()
    host = (p.hostname or "").lower()
    netloc = host
    if p.port:
        netloc = f"{host}:{p.port}"

    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    kept = [(k, v) for (k, v) in parse_qs(p.query, keep_blank_values=True).items()
            if k.lower() not in TRACKING_PARAMS]
    # parse_qs returns lists; flatten deterministically for a stable key.
    flat = []
    for k, vals in sorted(kept):
        for v in vals:
            flat.append((k, v))
    query = urlencode(flat)

    return urlunparse((scheme, netloc, path, "", query, ""))


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _parse_engine_date(raw):
    """
    Normalize a SearXNG publishedDate (ISO-ish string) to a tz-aware UTC datetime.
    Returns None if absent/unparseable.
    """
    if not raw:
        return None
    if isinstance(raw, dt.datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=dt.timezone.utc)
    s = str(raw).strip()
    if not s:
        return None
    # Handle trailing 'Z' (Zulu) which fromisoformat historically rejects.
    s = s.replace("Z", "+00:00")
    try:
        d = dt.datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s[:len(fmt) + 4], fmt).replace(
                tzinfo=dt.timezone.utc)
        except Exception:
            continue
    return None


def _searxng_page(query, pageno, time_range):
    """Fetch ONE SearXNG JSON page. Returns the raw results list (may be empty)."""
    try:
        import requests
        params = {
            "q": query,
            "format": "json",
            "categories": SEARXNG_CATEGORIES,
            "pageno": pageno,
        }
        if time_range:
            params["time_range"] = time_range
        resp = requests.get(
            SEARXNG_URL,
            params=params,
            headers={"User-Agent": "Mozilla/5.0 (ContextDistiller/16.0)"},
            timeout=FETCH_TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"SEARXNG: non-200 status {resp.status_code} (page {pageno})",
                  file=sys.stderr)
            return []
        return resp.json().get("results", []) or []
    except Exception as ex:
        print(f"SEARXNG ERROR (page {pageno}): {type(ex).__name__}: {ex}",
              file=sys.stderr)
        return []


def _accumulate_pass(query, time_range, seen_norm, seen_domains, url_records):
    """
    Run one paginated pass (A). Mutates seen_norm / seen_domains / url_records
    in place. Three termination guards: target domains, max pages, stall-break.
    """
    for pageno in range(1, MAX_PAGES_TO_ACCUMULATE + 1):
        if len(seen_domains) >= TARGET_DISTINCT_DOMAINS:
            break

        results = _searxng_page(query, pageno, time_range)
        if not results:
            # empty page -> nothing more to accumulate from this pass
            break

        added_this_page = 0
        for r in results:
            raw_url = r.get("url")
            if not raw_url:
                continue
            norm = normalize_url(raw_url)
            if norm in seen_norm:
                continue          # URL dedup BEFORE fetching (the normalization win)
            seen_norm.add(norm)
            added_this_page += 1

            domain = _domain_of(norm)
            seen_domains.add(domain)
            url_records.append({
                "url": raw_url,                       # fetch the original URL
                "domain": domain,
                "engine_date": _parse_engine_date(r.get("publishedDate")),
            })

        # Stall-break: a full page that added zero new unique URLs -> stop.
        if added_this_page == 0:
            print(f"STALL-BREAK: page {pageno} added 0 new URLs (tr={time_range!r}).",
                  file=sys.stderr)
            break


def fetch_searxng_index(query: str):
    """
    (A) Primary index: local SearXNG, paginated accumulator with dual recency pass.
    Returns a list of url_record dicts: {url, domain, engine_date}.
    """
    seen_norm = set()
    seen_domains = set()
    url_records = []

    # Pass 1: recency-biased (time_range). News engines honor this; others are
    # silently dropped by SearXNG, which is why pass 2 exists.
    _accumulate_pass(query, SEARXNG_TIME_RANGE, seen_norm, seen_domains, url_records)

    # Pass 2: unfiltered, to backfill evergreen/background sources and any
    # engines that don't support time_range. Merged into the same dedup sets.
    if DUAL_PASS_RECENCY and len(seen_domains) < TARGET_DISTINCT_DOMAINS:
        _accumulate_pass(query, "", seen_norm, seen_domains, url_records)

    return url_records


def fetch_ddg_index(query: str):
    """Fallback index source: DuckDuckGo HTML endpoint via the hardened fetch."""
    html_raw = execute_hardened_fetch(
        f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    )
    if not html_raw:
        return []
    soup = BeautifulSoup(html_raw, "html.parser")
    seen_norm = set()
    records = []
    for node in soup.select("a.result__url"):
        href = node.get("href", "")
        if "/l/?uddg=" in href:
            target = parse_qs(urlparse(href).query).get("uddg", [None])[0]
            if target:
                norm = normalize_url(target)
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                records.append({"url": target,
                                "domain": _domain_of(target),
                                "engine_date": None})  # DDG HTML gives no date
        if len(records) >= TARGET_DISTINCT_DOMAINS:
            break
    return records


def fetch_organic_index(query: str):
    """SearXNG primary (paginated), DuckDuckGo fallback. Returns url_records."""
    records = fetch_searxng_index(query)
    if records:
        return records
    print("SEARXNG returned no links; using DDG fallback.", file=sys.stderr)
    return fetch_ddg_index(query)

# ======================================================================
# (B + D) PROCESSING, SEGMENTATION, DATE CAPTURE
# ======================================================================

def compute_jaccard_overlap(a: str, b: str) -> float:
    w1 = set(re.findall(r"\w+", a.lower()))
    w2 = set(re.findall(r"\w+", b.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def _scraped_date(html_data):
    """
    (D) Fallback date from the page itself via trafilatura bare_extraction
    (plain extract() discards metadata). Returns tz-aware UTC datetime or None.
    """
    try:
        meta = trafilatura.bare_extraction(html_data, with_metadata=True)
    except Exception:
        return None
    if not meta:
        return None
    raw = meta.get("date") if isinstance(meta, dict) else getattr(meta, "date", None)
    return _parse_engine_date(raw)


def process_and_segment(html_data, record):
    """
    (B) Sliding-window segmentation, step=1 -> 2-sentence overlap, no len-2
    tail truncation (the slice shortens naturally at the end).
    (D) Binds the resolved date (engine preferred, scraped fallback) to each
    segment so the output can be sorted newest-first.
    """
    if not html_data:
        return []
    text = trafilatura.extract(html_data)
    if not text:
        return []

    # Resolve date: prefer the cleaner engine date, fall back to scraped.
    resolved_date = record.get("engine_date") or _scraped_date(html_data)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s) > 20]
    segments = []
    # step=1: 2-sentence overlap. Range over the full length; the final slices
    # sentences[i:i+3] truncate gracefully, so no closing content is dropped.
    for i in range(0, len(sentences), 1):
        chunk = " ".join(sentences[i:i + 3])
        if len(chunk) > 130:
            segments.append({
                "url": record["url"],
                "domain": record["domain"],
                "date": resolved_date,           # may be None -> trails in sort
                "clean_text": chunk,
            })
    return segments

# ======================================================================
# (C) RETRIEVAL + HYBRID-COVERAGE RERANKER
# ======================================================================

def run_deep_search_retrieval(query: str) -> str:
    records = fetch_organic_index(query)
    if not records:
        return "ERROR: Search layer could not resolve any result links."

    raw_pool = []
    # Concurrent fetch; execute_hardened_fetch is self-contained per URL
    # (own PoolManager, no shared state), so it is thread-safe. Parsing /
    # segmentation runs back on the main thread as each fetch completes.
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
        future_to_rec = {executor.submit(execute_hardened_fetch, rec["url"]): rec
                         for rec in records}
        for future in as_completed(future_to_rec):
            rec = future_to_rec[future]
            try:
                html = future.result()
            except Exception as ex:
                print(f"FETCH WORKER ERROR for '{rec['url']}': "
                      f"{type(ex).__name__}: {ex}", file=sys.stderr)
                continue
            raw_pool.extend(process_and_segment(html, rec))

    if not raw_pool:
        return "ERROR: No usable text content was extracted from result pages."

    # Stage 1: Bi-encoder pre-filter into the right-sized candidate pool.
    query_vec = bi_encoder.encode(query, convert_to_tensor=True)
    texts = [c["clean_text"] for c in raw_pool]
    chunk_vecs = bi_encoder.encode(texts, convert_to_tensor=True)
    bi_scores = util.cos_sim(query_vec, chunk_vecs)[0]
    for i, score in enumerate(bi_scores):
        raw_pool[i]["bi_score"] = float(score)
    raw_pool.sort(key=lambda x: x["bi_score"], reverse=True)
    top_candidates = raw_pool[:MAX_CANDIDATES_POOL]

    # Stage 2: Cross-encoder rerank -> sigmoid (overflow-guarded).
    pairs = [[query, c["clean_text"]] for c in top_candidates]
    logits = cross_encoder.predict(pairs)
    for i, logit in enumerate(logits):
        try:
            prob = 1.0 / (1.0 + math.exp(-float(logit)))
        except OverflowError:
            prob = 0.0 if float(logit) < 0 else 1.0
        top_candidates[i]["prob"] = prob
    top_candidates.sort(key=lambda x: x["prob"], reverse=True)

    # Stage 3: Hybrid-Coverage selection.
    #   - relative floor (top - margin) AND a small absolute floor so the
    #     junk filter still bites when the whole query matches poorly,
    #   - `break` ONLY on the floor (list is sorted desc; nothing better follows),
    #   - `continue` on domain-cap and dedup (never terminate the loop on them).
    top_prob = top_candidates[0]["prob"]
    relative_floor = max(0.0, top_prob - RELATIVE_CUTOFF_MARGIN)
    effective_floor = max(relative_floor, ABSOLUTE_PROB_FLOOR)

    selected = []
    dedup_registry = []
    domain_counts = {}
    for item in top_candidates:
        # Floor first: sorted descending, so once we drop below, we stop.
        if item["prob"] < effective_floor:
            break

        domain = item["domain"]
        if domain_counts.get(domain, 0) >= MAX_PER_DOMAIN:
            continue

        is_duplicate = False
        if len(item["clean_text"]) > DEDUP_MIN_LEN:
            for chosen in dedup_registry:
                if compute_jaccard_overlap(item["clean_text"], chosen) > JACCARD_DEDUP_THRESHOLD:
                    is_duplicate = True
                    break
        if is_duplicate:
            continue

        selected.append(item)
        dedup_registry.append(item["clean_text"])
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if len(selected) >= MAX_RESULTS:
            break

    if not selected:
        return "ERROR: No matches met the confidence floor."

    # (D) Chronological sort: newest-first, undated entries trail. Use a fixed
    # epoch sentinel (not None) so the sort key never mixes None with datetimes.
    EPOCH = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    selected.sort(key=lambda x: x["date"] or EPOCH, reverse=True)

    body_lines = ["# Search Results (newest-first)\n"]
    for item in selected:
        date_str = item["date"].date().isoformat() if item["date"] else "undated"
        body_lines.append(f"### {item['url']}\n*({date_str})* {item['clean_text']}\n")
    final_text = "\n".join(body_lines)

    # Honest token accounting: report the actual distilled token count and the
    # measured raw-extraction count it was distilled from. No invented denominator.
    distilled_tokens = len(TOKEN_ENCODER.encode(final_text))
    raw_tokens = sum(len(TOKEN_ENCODER.encode(c["clean_text"])) for c in raw_pool)
    if raw_tokens > 0:
        pct = (distilled_tokens / raw_tokens) * 100
        ratio_line = (f"- Output is ~{pct:.1f}% of the {raw_tokens} tokens "
                      f"extracted from result pages (rough heuristic, not a "
                      f"comparison to any external search tool).")
    else:
        ratio_line = "- Raw extraction token count unavailable."

    final_text += ("\n\n---\n**Token Report**\n"
                   f"- Distinct sources returned: {len(selected)}\n"
                   f"- Output tokens: {distilled_tokens}\n"
                   f"{ratio_line}")
    return final_text

# ======================================================================
# JSON-RPC 2.0 STDIO LOOP  (UNCHANGED from v15)
# ======================================================================

def serialize_and_emit(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def emit_error(code, message, req_id=None):
    serialize_and_emit({"jsonrpc": "2.0", "id": req_id,
                        "error": {"code": code, "message": message}})


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break

        req_id = None
        try:
            try:
                req = json.loads(line.strip())
            except Exception:
                emit_error(-32700, "Parse error: invalid JSON.")
                continue

            if not isinstance(req, dict):
                emit_error(-32600, "Invalid Request: payload must be an object.")
                continue

            req_id = req.get("id")
            method = req.get("method")

            # Notifications carry no id and expect no response.
            if req_id is None and isinstance(method, str) and method.startswith("notifications/"):
                continue

            if method == "initialize":
                serialize_and_emit({
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "DeepSearch", "version": "16.0.0"},
                    },
                })

            elif method == "tools/list":
                serialize_and_emit({
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "tools": [{
                            "name": "deep_search",
                            "description": "Local deep search: paginated multi-engine "
                                           "retrieval (top distinct domains), cross-encoder "
                                           "noise-filtering, newest-first chronological output.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string",
                                              "description": "Search query string"}
                                },
                                "required": ["query"],
                            },
                        }]
                    },
                })

            elif method == "tools/call":
                params = req.get("params", {})
                tool_name = params.get("name")
                if tool_name != "deep_search":
                    emit_error(-32601, f"Method not found: tool '{tool_name}'.", req_id)
                    continue
                query = params.get("arguments", {}).get("query", "").strip()
                if not query:
                    emit_error(-32602, "Invalid params: 'query' is required.", req_id)
                    continue
                result = run_deep_search_retrieval(query)
                serialize_and_emit({
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": result}]},
                })

            else:
                emit_error(-32601, f"Method not found: '{method}'.", req_id)

        except Exception as exc:
            print(f"RUNTIME ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            try:
                emit_error(-32603, f"Internal error: {exc}", req_id)
            except Exception:
                pass


if __name__ == "__main__":
    main()