#!/usr/bin/env python3
"""Live probe: does a running server scope radix prefix reuse by ``cache_salt`` /
``extra_key`` on /v1/responses?

Seven requests, each reading ``usage.input_tokens_details.cached_tokens``. One
fresh nonce opens the system message and the user turn of every request, so no
earlier traffic (this probe's earlier runs included) can share even one page.

  A1  salt A, turn 1                          cold        cached == 0
  A2  salt A, turn 2 (turn 1 + reply + user)  same salt   cached >= half of A1's input
  B1  salt B, A2's exact input                other salt  cached == 0
  U1  no salt, A2's exact input               unsalted    cached == 0  (salted entries invisible)
  U2  no salt, again                          unsalted    cached >= half (one shared namespace)
  X1  extra_key = A's value, A2's input       extra_key   cached == 0  (not the cache_salt namespace)
  X2  extra_key = A's value, again            extra_key   cached >= half

A2 / U2 / X2 are the positive controls: if the server reports no cached tokens
at all (no --enable-cache-report) they fail instead of letting the zeros pass.
The A and B requests carry a codex-style ``session_id`` that is closed at the end.

Needs only the standard library:

  python3 probe_responses_cache_salt.py --base-url http://127.0.0.1:30471

Exit status 0 when every row passes. With --session-ledger-all the server's TITO
ledger records these requests like any other.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid


def http(method: str, url: str, body: dict | None = None, timeout: float = 600.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url} -> HTTP {e.code}: {e.read()[:2000]!r}")
    return json.loads(raw) if raw.strip().startswith((b"{", b"[")) else raw.decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True, help="server root, e.g. http://127.0.0.1:30471")
    ap.add_argument("--model", help="served model name (default: the only one /v1/models lists)")
    ap.add_argument("--filler-lines", type=int, default=300,
                    help="lines of filler in the system message (~12 tokens each)")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    root = args.base_url.rstrip("/")
    if root.endswith("/v1"):
        raise SystemExit("--base-url is the server root, without /v1")

    models = [m["id"] for m in http("GET", root + "/v1/models", timeout=args.timeout)["data"]]
    model = args.model
    if model is None:
        if len(models) != 1:
            raise SystemExit(f"/v1/models lists {models}; pass --model")
        model = models[0]
    if model not in models:
        raise SystemExit(f"--model {model!r} is not served here: {models}")
    info = http("GET", root + "/server_info", timeout=args.timeout)
    print(f"server {root} model {model} page_size={info.get('page_size')} "
          f"enable_cache_report={info.get('enable_cache_report')} "
          f"disable_radix_cache={info.get('disable_radix_cache')}")

    nonce = uuid.uuid4().hex
    salt_a, salt_b = f"probe-{nonce}-A", f"probe-{nonce}-B"
    sid_a, sid_b = f"probe-{nonce}-A:0", f"probe-{nonce}-B:0"
    filler = "\n".join(
        f"Rule {i}: cell ({i % 30}, {(7 * i) % 30}) keeps colour {i % 10} unless rule {i + 1} applies."
        for i in range(args.filler_lines)
    )
    instructions = f"Probe {nonce}. Answer in one short sentence.\n{filler}"
    turn1 = [{"role": "user", "content": f"[{nonce}] Which colour does rule 3 give?"}]

    def respond(name, input_items, *, cache_salt=None, extra_key=None, session_id=None):
        body = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
            "max_output_tokens": 16,
            "temperature": 0,
            "store": False,
            "stream": False,
            "reasoning": {"effort": "none"},
        }
        if cache_salt is not None:
            body["cache_salt"] = cache_salt
        if extra_key is not None:
            body["extra_key"] = extra_key
        if session_id is not None:
            body["session_id"] = session_id
        out = http("POST", root + "/v1/responses", body, timeout=args.timeout)
        usage = out.get("usage")
        if not usage or "input_tokens_details" not in usage:
            raise SystemExit(f"{name}: response has no usage.input_tokens_details: {json.dumps(out)[:2000]}")
        text = "".join(
            part.get("text", "")
            for item in out.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
        )
        return usage["input_tokens"], usage["input_tokens_details"]["cached_tokens"], text

    rows = []

    def row(name, ns, n_in, cached, expect, ok):
        rows.append((name, ns, n_in, cached, expect, ok))

    n1, c1, reply = respond("A1", turn1, cache_salt=salt_a, session_id=sid_a)
    row("A1", "cache_salt=A", n1, c1, "== 0 (cold)", c1 == 0)
    turn2 = turn1 + [
        {"role": "assistant", "content": reply or "Colour 3."},
        {"role": "user", "content": "And rule 4?"},
    ]
    half = n1 // 2

    n, c, _ = respond("A2", turn2, cache_salt=salt_a, session_id=sid_a)
    row("A2", "cache_salt=A", n, c, f">= {half} (same episode)", c >= half)
    n, c, _ = respond("B1", turn2, cache_salt=salt_b, session_id=sid_b)
    row("B1", "cache_salt=B", n, c, "== 0 (other episode)", c == 0)
    n, c, _ = respond("U1", turn2)
    row("U1", "none", n, c, "== 0 (salted entries invisible)", c == 0)
    n, c, _ = respond("U2", turn2)
    row("U2", "none", n, c, f">= {half} (shared namespace)", c >= half)
    n, c, _ = respond("X1", turn2, extra_key=salt_a)
    row("X1", "extra_key=A", n, c, "== 0 (not cache_salt=A)", c == 0)
    n, c, _ = respond("X2", turn2, extra_key=salt_a)
    row("X2", "extra_key=A", n, c, f">= {half} (extra_key namespace)", c >= half)

    for sid in (sid_a, sid_b):
        http("POST", root + "/close_session", {"session_id": sid}, timeout=args.timeout)

    print(f"{'req':4} {'namespace':13} {'input':>6} {'cached':>6}  expectation")
    for name, ns, n_in, cached, expect, ok in rows:
        print(f"{name:4} {ns:13} {n_in:6d} {cached:6d}  {expect:36} {'PASS' if ok else 'FAIL'}")
    failed = [r[0] for r in rows if not r[5]]
    print("ALL PASS" if not failed else f"FAILED: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
