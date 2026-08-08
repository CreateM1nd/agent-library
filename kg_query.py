#!/usr/bin/env python3
"""Read side of the knowledge graph.

kg_extract.py fills rag_library_entities/rag_library_relations nightly; until
now nothing read them back, so the graph was unreachable from the agent. This
module is the query layer, and library_mcp_server.py exposes it as two tools.

Two things shape the design:

1. The write path normalises names before storing them in norm_name. Lookup
   MUST use the identical function or an exact match silently fails --
   "Active Directory" would not find the row stored under "active directory".
   Both sides import kg_ontology.normalize_name so there is one definition
   rather than two that have to stay in agreement forever.

2. Edges are stored per source file: the unique key is
   (from_id, rel, to_id, source_path), so one relation asserted by six
   different files is six rows. Returning those raw would show the same edge
   six times and bury genuinely distinct relations. Every traversal here
   aggregates -- summing mentions and counting distinct sources -- because
   "six files agree on this" is a confidence signal, not six facts.
"""
import sys

import psycopg

from db import DB_DSN
from kg_ontology import normalize_name as normalize

# An ambiguous lookup returns candidates instead of silently picking the most
# mentioned one. Guessing is worse than asking here: the agent can re-query
# with an exact name, but it cannot detect that it was handed the wrong entity.
MAX_CANDIDATES = 8
DEFAULT_RELATION_LIMIT = 20
DEFAULT_SOURCE_SAMPLE = 5


def _resolve(cur, name):
    """Return (entity_row, candidates). Exactly one of them is meaningful.

    Exact normalised match wins outright. Failing that, substring match ranked
    by mentions -- so "kerberos" finds "Kerberos Authentication" without the
    caller needing the full surface form."""
    norm = normalize(name)
    cur.execute(
        "SELECT id, name, type, mentions, source_paths FROM rag_library_entities WHERE norm_name = %s",
        (norm,),
    )
    row = cur.fetchone()
    if row:
        return row, []

    cur.execute(
        """
        SELECT id, name, type, mentions, source_paths
        FROM rag_library_entities
        WHERE norm_name LIKE %s
        ORDER BY mentions DESC
        LIMIT %s
        """,
        (f"%{norm}%", MAX_CANDIDATES),
    )
    candidates = cur.fetchall()
    if len(candidates) == 1:
        return candidates[0], []
    return None, candidates


def _candidate_payload(candidates):
    return {
        "found": False,
        "candidates": [
            {"name": c[1], "type": c[2], "mentions": c[3]} for c in candidates
        ],
    }


def lookup_entity(name, relation_limit=DEFAULT_RELATION_LIMIT):
    """One entity: its type, how often it was seen, which files mention it,
    and the relations it participates in on both sides."""
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        row, candidates = _resolve(cur, name)
        if row is None:
            return _candidate_payload(candidates)

        entity_id, ent_name, ent_type, mentions, source_paths = row

        cur.execute(
            """
            SELECT r.rel, e.name, e.type,
                   SUM(r.mentions) AS weight,
                   COUNT(DISTINCT r.source_path) AS sources,
                   'out' AS direction
            FROM rag_library_relations r
            JOIN rag_library_entities e ON e.id = r.to_id
            WHERE r.from_id = %s
            GROUP BY r.rel, e.name, e.type
            UNION ALL
            SELECT r.rel, e.name, e.type,
                   SUM(r.mentions), COUNT(DISTINCT r.source_path), 'in'
            FROM rag_library_relations r
            JOIN rag_library_entities e ON e.id = r.from_id
            WHERE r.to_id = %s
            GROUP BY r.rel, e.name, e.type
            ORDER BY 5 DESC, 4 DESC
            LIMIT %s
            """,
            (entity_id, entity_id, relation_limit),
        )
        relations = [
            {
                "relation": rel,
                "direction": direction,
                "entity": other,
                "entity_type": other_type,
                "sources": sources,
            }
            for rel, other, other_type, _weight, sources, direction in cur.fetchall()
        ]

    return {
        "found": True,
        "name": ent_name,
        "type": ent_type,
        "mentions": mentions,
        "source_count": len(source_paths or []),
        "sources": sorted(source_paths or [])[:DEFAULT_SOURCE_SAMPLE],
        "relations": relations,
    }


def related(name, relation=None, direction="both", limit=DEFAULT_RELATION_LIMIT):
    """Neighbours of one entity, optionally filtered to a single predicate.

    direction: "out" (name -> other), "in" (other -> name), or "both".
    Ordered by how many distinct files assert the edge, then total mentions --
    corroboration first, raw frequency second."""
    if direction not in ("out", "in", "both"):
        raise ValueError('direction must be "out", "in" or "both"')

    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        row, candidates = _resolve(cur, name)
        if row is None:
            return _candidate_payload(candidates)
        entity_id, ent_name = row[0], row[1]

        clauses = []
        params = []
        if direction in ("out", "both"):
            clauses.append(
                """
                SELECT r.rel, e.name, e.type, SUM(r.mentions), COUNT(DISTINCT r.source_path), 'out'
                FROM rag_library_relations r
                JOIN rag_library_entities e ON e.id = r.to_id
                WHERE r.from_id = %s AND (%s::text IS NULL OR r.rel = %s::text)
                GROUP BY r.rel, e.name, e.type
                """
            )
            params += [entity_id, relation, relation]
        if direction in ("in", "both"):
            clauses.append(
                """
                SELECT r.rel, e.name, e.type, SUM(r.mentions), COUNT(DISTINCT r.source_path), 'in'
                FROM rag_library_relations r
                JOIN rag_library_entities e ON e.id = r.from_id
                WHERE r.to_id = %s AND (%s::text IS NULL OR r.rel = %s::text)
                GROUP BY r.rel, e.name, e.type
                """
            )
            params += [entity_id, relation, relation]

        cur.execute(
            " UNION ALL ".join(clauses) + " ORDER BY 5 DESC, 4 DESC LIMIT %s",
            params + [limit],
        )
        rows = cur.fetchall()

    return {
        "found": True,
        "name": ent_name,
        "filter": {"relation": relation, "direction": direction},
        "results": [
            {
                "relation": rel,
                "direction": d,
                "entity": other,
                "entity_type": other_type,
                "sources": sources,
            }
            for rel, other, other_type, _weight, sources, d in rows
        ],
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python kg_query.py ENTITY [RELATION]", file=sys.stderr)
        return 1
    name = sys.argv[1]
    if len(sys.argv) > 2:
        payload = related(name, relation=sys.argv[2])
        rows = payload.get("results", [])
    else:
        payload = lookup_entity(name)
        rows = payload.get("relations", [])

    if not payload.get("found"):
        print(f"no exact match for {name!r}. candidates:")
        for c in payload["candidates"]:
            print(f"  {c['name']}  ({c['type']}, {c['mentions']} mentions)")
        return 1

    header = payload["name"]
    if payload.get("type"):
        header += f"  [{payload['type']}]"
    print(header)
    if "mentions" in payload:
        print(f"  {payload['mentions']} mentions across {payload['source_count']} files")
    if not rows:
        filt = payload.get("filter", {}).get("relation")
        print(f"  no {filt + ' ' if filt else ''}relations recorded")
    for r in rows:
        arrow = "->" if r["direction"] == "out" else "<-"
        print(f"  {arrow} {r['relation']:<12} {r['entity']}  ({r['entity_type']}, {r['sources']} sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
