#!/usr/bin/env python3
"""
Deep Search MCP Server (Version 15.0)
Local-first JSON-RPC stdio search tool with IP-pinned SSRF-hardened fetch,
streaming byte ceiling, cross-encoder reranking, and honest token accounting.

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
import tiktoken
from urllib.parse import quote_plus, urlparse, parse_qs, urljoin
from bs4 import BeautifulSoup
import trafilatura
from urllib3.poolmanager import PoolManager
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# ======================================================================
# GLOBAL CONFIGURATION
# ======================================================================
MAX_INDEX_LINKS = 20     #change it to 4,if needed
MAX_CANDIDATES_POOL = 120
MAX_PAGE_SIZE_BYTES = 512 * 1024        # Hard byte ceiling, enforced during streaming
MAX_RESULTS = 20                         # Final excerpts returned
MAX_REDIRECT_HOPS = 3
PROBABILITY_CUTOFF_MARGIN = 0.40        # Keep chunks within 0.15 of top probability
JACCARD_DEDUP_THRESHOLD = 0.35
DEDUP_MIN_LEN = 180                     # Only dedup long narrative blocks
FETCH_TIMEOUT = 5.0
TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

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
# SECURITY & FETCH ENGINE
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
                "User-Agent": "Mozilla/5.0 (ContextDistiller/15.0)",
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


def fetch_organic_index(query: str):
    """Retrieve result URLs via the DuckDuckGo HTML endpoint, decoding uddg redirects."""
    html_raw = execute_hardened_fetch(
        f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    )
    if not html_raw:
        return []
    soup = BeautifulSoup(html_raw, "html.parser")
    links = []
    for node in soup.select("a.result__url"):
        href = node.get("href", "")
        if "/l/?uddg=" in href:
            target = parse_qs(urlparse(href).query).get("uddg", [None])[0]
            if target:
                links.append(target)
    return links[:MAX_INDEX_LINKS]

# ======================================================================
# PROCESSING & RERANKING
# ======================================================================

def compute_jaccard_overlap(a: str, b: str) -> float:
    w1 = set(re.findall(r"\w+", a.lower()))
    w2 = set(re.findall(r"\w+", b.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def process_and_segment(html_data, url):
    """Sliding-window sentence segmentation (3 sentences, step 2 -> 1 overlap)."""
    if not html_data:
        return []
    text = trafilatura.extract(html_data)
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s) > 20]
    segments = []
    for i in range(0, len(sentences), 2):
        chunk = " ".join(sentences[i:i + 3])
        if len(chunk) > 130:
            segments.append({"url": url, "clean_text": chunk})
    return segments


def run_deep_search_retrieval(query: str) -> str:
    urls = fetch_organic_index(query)
    if not urls:
        return "ERROR: Search layer could not resolve any result links."

    raw_pool = []
    for link in urls:
        raw_pool.extend(process_and_segment(execute_hardened_fetch(link), link))

    if not raw_pool:
        return "ERROR: No usable text content was extracted from result pages."

    # Stage 1: Bi-encoder pre-filter
    query_vec = bi_encoder.encode(query, convert_to_tensor=True)
    texts = [c["clean_text"] for c in raw_pool]
    chunk_vecs = bi_encoder.encode(texts, convert_to_tensor=True)
    bi_scores = util.cos_sim(query_vec, chunk_vecs)[0]
    for i, score in enumerate(bi_scores):
        raw_pool[i]["bi_score"] = float(score)
    raw_pool.sort(key=lambda x: x["bi_score"], reverse=True)
    top_candidates = raw_pool[:MAX_CANDIDATES_POOL]

    # Stage 2: Cross-encoder rerank -> sigmoid (overflow-guarded)
    pairs = [[query, c["clean_text"]] for c in top_candidates]
    logits = cross_encoder.predict(pairs)
    for i, logit in enumerate(logits):
        try:
            prob = 1.0 / (1.0 + math.exp(-float(logit)))
        except OverflowError:
            prob = 0.0 if float(logit) < 0 else 1.0
        top_candidates[i]["prob"] = prob
    top_candidates.sort(key=lambda x: x["prob"], reverse=True)

    # Stage 3: Relative slicing + Jaccard dedup on long blocks
    probability_floor = max(0.0, top_candidates[0]["prob"] - PROBABILITY_CUTOFF_MARGIN)
    selected = []
    dedup_registry = []
    for item in top_candidates:
        if item["prob"] < probability_floor:
            break
        is_duplicate = False
        if len(item["clean_text"]) > DEDUP_MIN_LEN:
            for chosen in dedup_registry:
                if compute_jaccard_overlap(item["clean_text"], chosen) > JACCARD_DEDUP_THRESHOLD:
                    is_duplicate = True
                    break
        if not is_duplicate:
            selected.append(item)
            dedup_registry.append(item["clean_text"])
        if len(selected) >= MAX_RESULTS:
            break

    if not selected:
        return "ERROR: No matches met the relative confidence margin."

    body_lines = ["# Search Results\n"]
    for item in selected:
        body_lines.append(f"### {item['url']}\n{item['clean_text']}\n")
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
                   f"- Output tokens: {distilled_tokens}\n"
                   f"{ratio_line}")
    return final_text

# ======================================================================
# JSON-RPC 2.0 STDIO LOOP
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
                        "serverInfo": {"name": "DeepSearch", "version": "15.0.0"},
                    },
                })

            elif method == "tools/list":
                serialize_and_emit({
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "tools": [{
                            "name": "deep_search",
                            "description": "Local deep search with IP-pinned fetch, "
                                           "cross-encoder reranking, and token-trimmed output.",
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