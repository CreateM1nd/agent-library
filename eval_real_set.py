#!/usr/bin/env python3
"""Extract the queries Maria was actually asked, from her own trajectories.

The synthetic set (eval_build_set.py) measures one thing well and a different
thing not at all. Its questions are generated from the passage they should
retrieve, so they are full-sentence, natural-language, and share vocabulary
with their target. Real usage turns out to look almost nothing like that:

    14x  <title> <author>
    11x  upload files anonymously nextcloud
     8x  "CHAPTER" "<author>"

Mostly known-item lookups -- find this book, this chapter, this author -- often
with quoted phrases, often no verb at all. That is a different retrieval task,
and it is the one this corpus is actually used for, so it is the one worth
optimising. Measuring only the synthetic set would tune the system for a
workload nobody generates.

There is no gold label here: nobody recorded which passage was "right" for
"The Art Of War". Labels come from eval_judge_pool.py, which pools every
configuration's candidates and judges them, so no configuration is favoured by
the labelling.
"""
import argparse
import collections
import glob
import os
import json
import sys
from pathlib import Path

SESSIONS = os.environ.get("AGENT_SESSIONS_GLOB",
                          "~/.openclaw/agents/main/sessions/*.trajectory.jsonl")
OUT_DIR = Path(__file__).parent / "eval"
MIN_QUERY_CHARS = 3


def extract():
    """Distinct query -> number of times it was actually issued.

    Trajectories store a tool call as {"type":"toolCall", "name":
    "library-search__library_search", "arguments":{"query":...}} -- note the
    MCP server prefix on the name, and `arguments` rather than `args`/`input`.
    Guessing that shape wrong returns zero matches silently, which is exactly
    what happened on the first attempt.
    """
    counts = collections.Counter()
    for path in glob.glob(os.path.expanduser(SESSIONS)):
        for line in open(path, errors="ignore"):
            if "library_search" not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in event.get("data", {}).get("messagesSnapshot", []) or []:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "toolCall":
                        continue
                    if "library_search" not in str(block.get("name", "")):
                        continue
                    query = (block.get("arguments") or {}).get("query")
                    if not isinstance(query, str):
                        continue
                    query = query.strip()
                    if len(query) < MIN_QUERY_CHARS:
                        continue
                    # OpenClaw's own secret-redaction fired on some stored
                    # trajectory entries -- the same title appears as both
                    # "Basic Mathematics" and "Basic <redacted>". A query
                    # containing the literal placeholder searches for nothing
                    # real, so it would measure noise rather than retrieval.
                    if "<redacted>" in query.lower():
                        continue
                    counts[query] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="Extract real library_search queries")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else OUT_DIR / "real-queries.jsonl"

    counts = extract()
    if not counts:
        print("no library_search calls found in trajectories", file=sys.stderr)
        return 1

    # Frequency is kept, not used for weighting: a query issued 14 times was
    # 14 attempts at one unmet need, which says the retrieval failed, not that
    # the query matters 14x more. It is recorded so that signal is not lost.
    with out_path.open("w") as f:
        for query, n in counts.most_common():
            f.write(json.dumps({"query": query, "times_issued": n}) + "\n")

    print(f"{len(counts)} distinct queries ({sum(counts.values())} calls) -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
