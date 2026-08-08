#!/usr/bin/env python3
import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
# Defaults to the same local chat model the rest of the stack uses. Overridable because bulk
# summarization is a batch job, not live reasoning -- a smaller model can be
# several times faster at corpus scale if its summary quality holds up.
CHAT_MODEL = os.environ.get("RAPTOR_CHAT_MODEL", "qwen3.6:latest")


def summarize(prompt, system=None):
    """One-shot, non-streaming chat call with thinking disabled -- verified
    against the live model that think:false cuts a trivial call from ~3.1s
    to ~0.3s (thinking-trace generation is the dominant cost for a model
    this size). RAPTOR cluster summarization doesn't need chain-of-thought,
    just concise synthesis, so this is a real win at the call volumes a
    multi-hundred-book corpus implies, not a premature optimization."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"].strip()
