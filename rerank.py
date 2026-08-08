#!/usr/bin/env python3
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query, candidates):
    """Second-stage reranking: unlike vector/keyword search, which score
    each candidate against the query independently, a cross-encoder reads
    the query and each candidate TOGETHER, so it can judge real relevance
    instead of relying on embedding-space distance. candidates is a list
    of text strings; returns the same-length list of scores in the same
    order (higher = more relevant)."""
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, text) for text in candidates]
    return model.predict(pairs).tolist()
