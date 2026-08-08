#!/usr/bin/env python3
"""Closed vocabulary for the library knowledge graph.

Modelled on memory-wiki's WIKI_PAGE_KINDS: a fixed tuple enforced in code, not
a suggestion in a prompt. The first extraction pass proved why that distinction
matters -- the prompt named example types and the model still produced 175
distinct types and 650 distinct predicates over 93 books, including `tool` (980)
alongside `tools` (325), and `part of` (71) alongside `part_of` (42). Splitting
one concept across several labels silently breaks traversal: a query filtering
type='tool' misses a quarter of the tools and returns a confidently incomplete
answer rather than an error.

Everything here is deliberately small. The point is not to describe the domain
richly -- embeddings already handle nuance -- it is to make traversal reliable.
Anything unmapped becomes "other" rather than inventing a new bucket, so drift
cannot creep back in one night at a time.
"""
import re

ENTITY_TYPES = (
    "tool",           # named software an operator runs (Burp Suite, sqlmap, Metasploit)
    "library",        # code a developer imports (Django, Scapy, React)
    "protocol",       # wire/interop protocols and formats (HTTP, LLMNR, PCAP)
    "vulnerability",  # named vulns and attack classes (Cross-Site Scripting, EternalBlue)
    "technique",      # named methods (Pass-the-Hash, threat modelling)
    "standard",       # specs, frameworks, certifications (PCI-DSS, NIST 800-53, OSCP)
    "organization",   # companies, projects, bodies (OWASP, MITRE, Microsoft)
    "person",         # named individuals
    "concept",        # domain ideas that are not any of the above
    "platform",       # operating systems and runtimes (Kali Linux, Android, Node.js)
    "service",        # hosted/network services (AWS S3, Active Directory)
    "file_format",    # (PDF, GLB, JSON)
    "resource",       # books, courses, documents, URLs -- things you go and read
    "other",          # recognised entity, no confident bucket
)

RELATION_TYPES = (
    "uses",           # A operates/invokes B
    "part_of",        # A is a component of B
    "type_of",        # A is a kind of B  (is_a, is a type of)
    "alias_of",       # A is another name for B (abbreviation, also known as, same_as)
    "targets",        # A attacks/exploits/affects B
    "implements",     # A provides an implementation of B
    "requires",       # A depends on B
    "runs_on",        # A executes on platform B
    "created_by",     # A was authored/published by B
    "mitigates",      # A defends against B
    "related_to",     # recognised link, no confident predicate
)

# Longest-match-first: "named vulnerability or attack class" must be tested
# before "vulnerability", or the substring rule would mis-bucket it.
_ENTITY_RULES = (
    ("vulnerability", ("vulnerab", "attack class", "attack_class", "exploit", "cve", "weakness")),
    ("library", ("library", "libraries", "framework", "package", "module", "sdk", "api")),
    ("protocol", ("protocol", "port", "network address", "encoding", "cipher", "algorithm")),
    ("technique", ("technique", "method", "tactic", "procedure", "attack vector", "practice")),
    ("standard", ("standard", "specification", "certification", "cert", "compliance",
                  "regulation", "policy", "benchmark", "model", "law")),
    ("organization", ("organization", "organisation", "company", "vendor", "agency",
                      "institution", "team", "group", "project", "community")),
    ("person", ("person", "people", "author", "researcher", "individual", "role")),
    ("platform", ("platform", "operating system", "os", "distro", "distribution",
                  "runtime", "environment", "device", "hardware")),
    ("service", ("service", "server", "cloud", "database", "datastore", "provider")),
    ("file_format", ("file format", "file_format", "format", "extension", "filetype")),
    ("resource", ("book", "course", "document", "paper", "article", "url", "website",
                  "publication", "report", "guide", "reference", "work", "event")),
    ("tool", ("tool", "utility", "program", "software", "application", "app",
              "command", "binary", "suite", "scanner", "product", "technology")),
    ("concept", ("concept", "idea", "principle", "term", "category", "topic", "field")),
)

_RELATION_RULES = (
    ("alias_of", ("alias", "abbreviation", "also known as", "also_known_as", "same_as",
                  "same as", "aka", "stands for", "short for", "synonym", "identifier")),
    ("type_of", ("is_a", "is a", "type of", "type_of", "kind of", "instance of",
                 "subclass", "category of", "classified")),
    ("part_of", ("part of", "part_of", "component of", "belongs to", "contained in",
                 "includes", "contains", "consists", "comprises", "has_")),
    ("targets", ("target", "attack", "exploit", "affect", "compromise", "against",
                 "vulnerable", "abuse", "crack", "bypass", "evade", "intercept",
                 "sniff", "brute", "hijack", "spoof", "inject")),
    ("mitigates", ("mitigat", "protect", "defend", "prevent", "block", "address",
                   "remediat", "fix", "secure")),
    ("implements", ("implement", "provides", "supports", "offers", "enables", "exposes")),
    ("requires", ("require", "depends", "needs", "prerequisite")),
    ("runs_on", ("runs on", "runs_on", "hosted", "deployed", "executes on", "installed")),
    ("created_by", ("created", "author", "developed", "written", "published",
                    "maintained", "made by", "founded")),
    ("uses", ("uses", "used by", "used_by", "used with", "utilis", "utiliz",
              "invokes", "calls", "leverages", "employs", "works with")),
)


def _normalize(raw):
    return re.sub(r"[^a-z0-9 ]+", " ", (raw or "").lower()).strip()


def normalize_name(name):
    """Case/punctuation-insensitive key used for exact entity matching.
    Deliberately conservative: it collapses spacing and punctuation noise but
    never stems or drops words, so 'Kali Linux' and 'Kali' stay distinct.

    Lives here rather than in kg_extract because the read path (kg_query) needs
    the identical function -- this value is what norm_name holds, so a lookup
    normalising even slightly differently silently misses instead of erroring.
    Two copies that must agree forever is exactly the drift this module exists
    to prevent."""
    n = name.lower().strip()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def canonical_entity_type(raw):
    """Map a free-text type onto ENTITY_TYPES. Unrecognised input becomes
    "other" rather than a new bucket -- an unmapped value is a vocabulary gap to
    review, not a licence to grow the type list at runtime."""
    n = _normalize(raw)
    if not n:
        return "other"
    for canon, needles in _ENTITY_RULES:
        if any(needle in n for needle in needles):
            return canon
    return "other"


def canonical_relation(raw):
    """Map a free-text predicate onto RELATION_TYPES, defaulting to related_to.

    Direction is NOT inferred here. "used by" and "uses" both map to `uses`
    even though they are converses; the extractor emits (from, rel, to) in
    reading order and rewriting direction from a phrase would silently invert
    edges. Preserving a possibly-reversed edge is recoverable; inverting one
    is not."""
    n = _normalize(raw)
    if not n:
        return "related_to"
    for canon, needles in _RELATION_RULES:
        if any(needle in n for needle in needles):
            return canon
    return "related_to"
