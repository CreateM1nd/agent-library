#!/usr/bin/env python3
import json
import os
import urllib.request

# Uses Ollama's /api/embed (not the legacy /api/embeddings), because only the
# newer endpoint honours a top-level "dimensions" parameter. qwen3-embedding
# natively emits 1024 dims; Matryoshka truncation to 768 keeps the existing
# rag_library_chunks vector(768) column and its HNSW index unchanged, avoiding
# a schema migration and index rebuild.
#
# Switched from nomic-embed-text (768 dims, 2048-token context) because that
# context ceiling was a real, observed failure: dotted-leader table-of-contents
# pages tokenize inefficiently and overflowed it, which is why chunk_text.py
# still caps structural chunks at MAX_STRUCTURAL_CHUNK_CHARS. qwen3-embedding
# has a 32K context, so that ceiling stops being the binding constraint.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "768"))


def embed(text):
    payload = {"model": EMBED_MODEL, "input": text}
    if EMBED_DIMENSIONS:
        payload["dimensions"] = EMBED_DIMENSIONS
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    # /api/embed returns {"embeddings": [[...]]}; the legacy /api/embeddings
    # returned {"embedding": [...]}. Accept either so the client still works if
    # OLLAMA_URL is pointed back at the old endpoint.
    if "embeddings" in body:
        return [float(x) for x in body["embeddings"][0]]
    return [float(x) for x in body["embedding"]]
