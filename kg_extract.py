#!/usr/bin/env python3
"""Incrementally build a knowledge graph over the library, a few books at a time.

Runs as a nightly job so the full corpus is covered gradually instead of in one
multi-day batch that would contend with the live agent for the GPU the whole time. Each
run takes a chunk budget rather than a book count: book sizes here range from a
handful of chunks to 1829, so "5 books" can mean 2 hours or 12 depending on
which five come up, while a chunk budget keeps every night roughly equal.

Entity resolution is what makes the incremental approach produce one graph
instead of N disconnected per-book graphs -- "Metasploit", "MSF" and "the
Metasploit Framework" have to land on one node or nothing can be traversed
across books. Matching is deliberately cheap first (normalized exact match,
then bounded edit distance) and only escalates from there, mirroring the
prefilter-before-LLM strategy in RAGflow's rag/graphrag/entity_resolution.py.

State lives entirely in the database (rag_library_kg_progress), so an
interrupted run resumes correctly and the corpus itself is the source of truth
about what is done.
"""
import argparse
import json
import re
import sys
import time

import psycopg
from pgvector.psycopg import register_vector

from embed_client import embed
from kg_ontology import (ENTITY_TYPES, RELATION_TYPES, canonical_entity_type,
                         canonical_relation, normalize_name as normalize)
from summarize_client import summarize

from db import DB_DSN

DEFAULT_CHUNK_BUDGET = 1400  # ~1.9h at the measured ~4.9s/chunk
EDIT_DISTANCE_MAX = 2  # only collapse near-identical surface forms; see _find_existing
MIN_NAME_LEN = 2
MAX_NAME_LEN = 80
TOP_MATCH_CHUNKS = 5  # chunks averaged when scoring a book against a topic

EXTRACT_SYSTEM_PROMPT = (
    "Extract named technical entities and their relationships from the text.\n"
    "\n"
    "INCLUDE only things with a proper name that someone could look up:\n"
    "  tools (Metasploit, Burp Suite), protocols (LLMNR, Kerberos), standards\n"
    "  (PCI-DSS, NIST 800-53), named vulnerabilities or attack classes\n"
    "  (Cross-Site Scripting, EternalBlue, CVE-2021-44228), named techniques\n"
    "  (Pass-the-Hash), libraries/frameworks (Django, Scapy), file formats\n"
    "  (PCAP), organizations (OWASP, MITRE).\n"
    "\n"
    "EXCLUDE, always:\n"
    "  programming language syntax, keywords and operators (==, and, elif,\n"
    "  while loop, for loop), built-in functions and types (len, print, list,\n"
    "  dict, input()), ordinary shell commands (cp, ls, cd), generic nouns\n"
    "  (file, user, server, password, network), chapter or section headings,\n"
    "  and any phrase that is simply prose rather than a nameable thing.\n"
    "\n"
    "Prefer the full canonical name over an abbreviation when the text gives\n"
    "both (write 'Cross-Site Scripting', not 'XSS'), so the same concept is\n"
    "not recorded under two names.\n"
    "\n"
    "If a chunk contains no such entities, return empty lists -- that is a\n"
    "correct answer, not a failure.\n"
    "\n"
    "Use ONLY these entity types:\n  " + ", ".join(ENTITY_TYPES) + "\n"
    "Use ONLY these relation values:\n  " + ", ".join(RELATION_TYPES) + "\n"
    "If nothing fits, use 'other' for a type and 'related_to' for a relation.\n"
    "\n"
    'Output compact JSON only: {"entities":[{"name":..,"type":..}],'
    '"relations":[{"from":..,"rel":..,"to":..}]}. '
    "Use the exact entity names in relations. No prose, no markdown fences."
)





def edit_distance_within(a, b, limit):
    """Bounded Levenshtein: returns True when a and b differ by <= limit edits.
    Length pre-check makes the common (very different) case free."""
    if abs(len(a) - len(b)) > limit:
        return False
    if a == b:
        return True
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


def parse_extraction(raw):
    """LLM output to (entities, relations). Tolerates fenced code blocks and
    trailing prose, which a 36B model still emits occasionally despite the
    prompt; a malformed response costs one chunk, never the whole run."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return [], []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return [], []
    ents, rels = [], []
    # Canonicalise here, at the boundary where model output enters the system,
    # so nothing downstream ever sees a free-text type or predicate. The prompt
    # below also names the allowed values, but the prompt is a request and this
    # is the enforcement -- the first pass proved the model drifts regardless of
    # what the prompt asks for.
    for e in data.get("entities", []) or []:
        name = str(e.get("name", "")).strip()
        if MIN_NAME_LEN <= len(name) <= MAX_NAME_LEN:
            ents.append((name, canonical_entity_type(e.get("type"))))
    for r in data.get("relations", []) or []:
        f, rel, t = (str(r.get(k, "")).strip() for k in ("from", "rel", "to"))
        if f and t and rel and len(f) <= MAX_NAME_LEN and len(t) <= MAX_NAME_LEN:
            rels.append((f, canonical_relation(rel), t))
    return ents, rels


def _find_existing(cur, name, cache):
    """Resolve a name to an existing entity id, or None.

    Two tiers, cheapest first: exact normalized match, then bounded edit
    distance against entities sharing a first character. The edit-distance tier
    exists for real variants seen in this corpus ('Metasploit'/'Metasploit',
    plural/singular), not for semantic aliases -- 'MSF' will not merge into
    'Metasploit' this way, and that is intentional for now, since guessing at
    semantic equivalence without an LLM judge produces confident wrong merges
    that are far harder to notice than duplicates."""
    norm = normalize(name)
    if not norm:
        return None
    if norm in cache:
        return cache[norm]
    cur.execute("SELECT id FROM rag_library_entities WHERE norm_name = %s", (norm,))
    row = cur.fetchone()
    if row:
        cache[norm] = row[0]
        return row[0]
    cur.execute(
        "SELECT id, norm_name FROM rag_library_entities WHERE left(norm_name,1) = left(%s,1) AND abs(length(norm_name) - length(%s)) <= %s",
        (norm, norm, EDIT_DISTANCE_MAX),
    )
    for eid, existing in cur.fetchall():
        if edit_distance_within(norm, existing, EDIT_DISTANCE_MAX):
            cache[norm] = eid
            return eid

    # Abbreviation matching by initials was tried here and deliberately removed.
    # It fails the case that motivated it -- "Cross-Site Scripting" initializes
    # to "css", not "xss", because XSS abbreviates "cross" as X -- and worse, it
    # would then merge Cross-Site Scripting into Cascading Style Sheets. A wrong
    # merge is far more damaging than a duplicate: duplicates can be merged later,
    # but a corrupted node silently poisons every traversal through it. Splitting
    # is instead reduced at the source, by having the extraction prompt prefer the
    # full canonical name over an abbreviation.
    return None


def upsert_entity(cur, name, etype, source_path, cache):
    eid = _find_existing(cur, name, cache)
    if eid is None:
        cur.execute(
            """
            INSERT INTO rag_library_entities (name, norm_name, type, mentions, source_paths)
            VALUES (%s, %s, %s, 1, ARRAY[%s])
            ON CONFLICT (norm_name) DO UPDATE SET mentions = rag_library_entities.mentions + 1
            RETURNING id
            """,
            (name, normalize(name), etype or None, source_path),
        )
        eid = cur.fetchone()[0]
        cache[normalize(name)] = eid
        return eid
    cur.execute(
        """
        UPDATE rag_library_entities
        SET mentions = mentions + 1,
            source_paths = CASE WHEN %s = ANY(source_paths) THEN source_paths
                                ELSE array_append(source_paths, %s) END,
            type = COALESCE(type, %s),
            updated_at = now()
        WHERE id = %s
        """,
        (source_path, source_path, etype or None, eid),
    )
    return eid


def process_book(conn, source_path, chunk_limit=None):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM rag_library_chunks WHERE source_path = %s AND summary_level = 0 ORDER BY chunk_index",
            (source_path,),
        )
        chunks = [r[0] for r in cur.fetchall()]
    if chunk_limit:
        chunks = chunks[:chunk_limit]

    cache = {}
    entities_seen = 0
    for chunk in chunks:
        try:
            raw = summarize(f"Text:\n\n{chunk}", system=EXTRACT_SYSTEM_PROMPT)
        except Exception as e:
            print(f"    chunk skipped ({type(e).__name__})", flush=True)
            continue
        ents, rels = parse_extraction(raw)
        if not ents:
            continue
        with conn.cursor() as cur:
            ids = {}
            for name, etype in ents:
                ids[normalize(name)] = upsert_entity(cur, name, etype, source_path, cache)
                entities_seen += 1
            for f, rel, t in rels:
                fid, tid = ids.get(normalize(f)), ids.get(normalize(t))
                if not fid or not tid or fid == tid:
                    continue
                cur.execute(
                    """
                    INSERT INTO rag_library_relations (from_id, rel, to_id, source_path)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (from_id, rel, to_id, source_path)
                    DO UPDATE SET mentions = rag_library_relations.mentions + 1
                    """,
                    (fid, rel, tid, source_path),
                )
        conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rag_library_kg_progress (source_path, chunks_done, entities_found)
            VALUES (%s, %s, %s)
            ON CONFLICT (source_path) DO UPDATE
              SET chunks_done = EXCLUDED.chunks_done,
                  entities_found = EXCLUDED.entities_found,
                  completed_at = now()
            """,
            (source_path, len(chunks), entities_seen),
        )
    conn.commit()
    return len(chunks), entities_seen


def pending_books(cur):
    cur.execute(
        """
        SELECT c.source_path, count(*) AS n
        FROM rag_library_chunks c
        WHERE c.summary_level = 0
          AND c.source_path NOT IN (SELECT source_path FROM rag_library_kg_progress)
        GROUP BY c.source_path
        ORDER BY n
        """
    )
    return cur.fetchall()


def pending_books_for_topic(cur, topic, vector):
    """Order pending books by how well they match a topic, so asking about a
    subject pulls the relevant unread books to the front of the queue.

    Scored against each book's highest-level RAPTOR summary, not its raw chunks.
    Both chunk-based rankings were tried and measured worse on real topics:
    single-best-chunk let a 9-chunk AI-tools PDF outrank the actual Active
    Directory book on one lucky match, and a top-5 mean dropped that book out
    of the top four entirely. A chunk answers "does this page mention the
    topic"; the top summary answers "is this book about the topic", which is
    the question being asked here.

    Books with no summary are excluded from topic ranking entirely rather than
    falling back to a best-chunk score: they are below RAPTOR's clustering
    minimum, so they are a handful of chunks that cost almost nothing and get
    picked up quickly by the default smallest-first order anyway -- while their
    single-chunk scores are noisy enough to outrank genuinely on-topic books
    (a 3-chunk checklist beat a 317-chunk Active Directory title)."""
    cur.execute(
        """
        WITH pending AS (
            SELECT source_path, count(*) AS n
            FROM rag_library_chunks
            WHERE summary_level = 0
              AND source_path NOT IN (SELECT source_path FROM rag_library_kg_progress)
            GROUP BY source_path
        ),
        top_summary AS (
            SELECT DISTINCT ON (source_path) source_path, embedding
            FROM rag_library_chunks
            WHERE summary_level > 0
            ORDER BY source_path, summary_level DESC
        )
        SELECT p.source_path, p.n, 1 - (s.embedding <=> %s::vector) AS score
        FROM pending p
        JOIN top_summary s ON s.source_path = p.source_path
        ORDER BY score DESC
        """,
        (vector,),
    )
    return [(path, n) for path, n, _ in cur.fetchall()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("books", nargs="*", help="specific books; default: next pending by chunk budget")
    parser.add_argument("--chunk-budget", type=int, default=DEFAULT_CHUNK_BUDGET,
                        help="stop starting new books once this many chunks are queued")
    parser.add_argument("--max-chunks-per-book", type=int, help="cap per book (pilot runs)")
    parser.add_argument("--topic", help="prioritize unread books matching this topic instead of smallest-first")
    args = parser.parse_args()

    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if args.topic:
                pending = pending_books_for_topic(cur, args.topic, embed(args.topic))
            else:
                pending = pending_books(cur)

    if args.books:
        selected = [(b, 0) for b in args.books]
    else:
        selected, total = [], 0
        for path, n in pending:
            if selected and total + n > args.chunk_budget:
                break
            selected.append((path, n))
            total += n

    if not selected:
        print("nothing pending", flush=True)
        return

    print(f"{len(pending)} books pending; this run: {len(selected)}", flush=True)
    started = time.time()
    with psycopg.connect(DB_DSN) as conn:
        for i, (path, n) in enumerate(selected, 1):
            t0 = time.time()
            print(f"[{i}/{len(selected)}] {path} ({n} chunks)", flush=True)
            try:
                done, ents = process_book(conn, path, args.max_chunks_per_book)
                print(f"    {done} chunks -> {ents} entity mentions ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"    FAILED: {e}", file=sys.stderr, flush=True)
    print(f"\ndone in {(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
