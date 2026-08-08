#!/usr/bin/env python3
import os

from mcp.server.mcpserver import MCPServer

from kg_query import lookup_entity, related
from search import find_files, search

mcp = MCPServer("library-search")


@mcp.tool()
def library_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the local document corpus by meaning, not keywords.

    Returns the top matching passages, each with its source file, approximate
    location, and a relevance score.

    The score is a cross-encoder reranking score, NOT a 0-1 similarity or a
    percentage: it is an unbounded logit where higher means more relevant, and
    values above 1 are normal. Compare scores against each other within one
    result set; do not read a score as a confidence level or a match
    percentage.

    Results may include passages taken verbatim from a book, or higher-level
    summaries synthesized from many passages of the same book. A summary is
    useful for a broad question about a file's overall treatment of a topic; a
    verbatim passage is what to cite for a specific claim."""
    results = search(query, top_k=top_k)
    return [
        {
            "source": source_path,
            "location": page_index,
            "kind": kind,
            "relevance": round(score, 4),
            "text": content,
        }
        for source_path, page_index, content, score, kind in results
    ]


@mcp.tool()
def library_files(pattern: str = "", limit: int = 30) -> dict:
    """Check whether the library CONTAINS a given work, by filename.

    Use this BEFORE claiming the library does or does not have something. This
    is the inventory; library_search is the contents. Searching for a work that
    is absent returns the closest available passages and gives no signal that
    it is missing, so "I don't have that" must come from here, never from
    library_search returning poor results.

    Returns {"total", "matched", "returned", "truncated", "files"}. A "matched"
    of 0 is a definitive negative -- report it plainly rather than offering the
    nearest topic.

    When "truncated" is true, "files" is a SAMPLE and not the whole answer. Do
    not describe a truncated list as complete, and do not infer the contents of
    the corpus from search results either: searching several topics returns
    examples, never an inventory. If a full listing is wanted, raise `limit`.

    Matching is loose on purpose (substring or stemmed word overlap), so a
    half-remembered title or wrong punctuation still finds a real file. An
    empty pattern returns the total and a sample.
    """
    return find_files(pattern, limit=limit)


@mcp.tool()
def library_entity(name: str) -> dict:
    """Look up one concept in the library's knowledge graph: what kind of thing
    it is, how many files discuss it, and what it connects to.

    Use this when the question is about a THING and its connections ("what is
    Kerberos related to", "what does Metasploit depend on") rather than about
    what a text says. For prose, quotes, or explanations, use library_search --
    this returns structure, not passages.

    If the name is ambiguous, returns {"found": false, "candidates": [...]}
    instead of guessing. Re-query with one of the candidate names.

    Coverage is partial: the graph is built incrementally and currently spans
    about 40% of the corpus, so an absent entity means "not extracted yet", not
    "not in the library". Extraction is model-generated and imperfect -- an
    occasional edge is miscategorised or reversed. Treat a relation as a lead
    worth confirming with library_search, not as a verified fact."""
    return lookup_entity(name)


@mcp.tool()
def library_related(name: str, relation: str = None, direction: str = "both",
                    limit: int = 20) -> dict:
    """Walk the knowledge graph outward from one concept, optionally along a
    single kind of edge.

    relation is one of: uses, part_of, type_of, alias_of, targets, implements,
    requires, runs_on, created_by, mitigates, related_to. Omit it to get every
    edge. direction is "out" (name -> other), "in" (other -> name), or "both".

    Results are ordered by how many distinct files assert the edge, so a
    relation backed by several sources ranks above one mentioned once. The
    "sources" count is that corroboration figure -- treat a 1 as weak.

    Same caveats as library_entity: partial coverage, model-generated edges."""
    return related(name, relation=relation, direction=direction, limit=limit)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
