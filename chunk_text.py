#!/usr/bin/env python3
import re
import sys
from pathlib import Path

from extract_text import extractor_for

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MAX_STRUCTURAL_CHUNK_CHARS = 1200  # verified too high at 3000: a dotted-leader-heavy
# TOC chunk that size overflowed nomic-embed-text's 2048-token context (dots
# tokenize inefficiently), causing a real embedding failure on 24 real books

TOC_HEADER_RE = re.compile(r"^\s*(table of )?contents\s*$", re.IGNORECASE)
CHAPTER_LINE_RE = re.compile(r"^\s*(chapter|part)\s+[\divxlc]+\b", re.IGNORECASE)
DOTTED_LEADER_RE = re.compile(r"\.{2,}\s*\d+\s*$")  # "Some Title . . . . 42"
BARE_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")


def is_structural_page(page_text, density_threshold=0.3):
    """A page 'looks like' a table of contents rather than prose: an
    explicit header near the top, or a high density of chapter/part
    markers, dotted-leader page references, and bare page numbers.

    Deliberately not applied book-wide -- verified against a real book
    (a textbook) that an answer-key section deep in
    the back matter produces the same "Chapter N" + short-line, number-
    heavy signature as the real table of contents (both are literally
    organized by chapter/section headers). Density alone can't tell them
    apart; where the page sits in the book can, since a real TOC is
    reliably near the front and an answer key isn't -- see the
    front_matter_cutoff restriction in chunk_pages."""
    lines = [l for l in page_text.splitlines() if l.strip()]
    if not lines:
        return False
    if any(TOC_HEADER_RE.match(l) for l in lines[:3]):
        return True
    structural_lines = sum(
        1
        for l in lines
        if CHAPTER_LINE_RE.match(l) or DOTTED_LEADER_RE.search(l) or BARE_PAGE_NUM_RE.match(l)
    )
    return (structural_lines / len(lines)) > density_threshold


def _fixed_size_chunks(page_slice, chunk_size, overlap):
    """Fixed-size sliding-window chunking over a contiguous run of
    (page_index, page_text) pairs. Same algorithm as before, just scoped
    to whatever prose run it's handed rather than the whole book."""
    full_text = ""
    offsets = []
    for page_index, page_text in page_slice:
        offsets.append((len(full_text), page_index))
        full_text += page_text + "\n"

    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        text = full_text[start:end].strip()
        if text:
            page_index = page_slice[0][0]
            for offset, idx in offsets:
                if offset <= start:
                    page_index = idx
                else:
                    break
            chunks.append({"text": text, "page_index": page_index, "chunk_type": "prose"})
        if end == len(full_text):
            break
        start = end - overlap
    return chunks


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Structure-aware chunking: contiguous runs of table-of-contents-like
    pages are kept together as their own chunk(s) instead of being sliced
    apart by the fixed-size window, since a TOC split across chunk
    boundaries stops being a usable, coherent answer to "what are the
    chapters". Everything else still uses the original fixed-size
    sliding-window approach."""
    # Real tables of contents sit in the front matter; verified against a real
    # book that scanning the whole document for the same density signal also
    # catches an answer-key section with an identical "Chapter N" signature.
    front_matter_cutoff = max(20, int(len(pages) * 0.15))

    chunks = []
    prose_run = []  # list of (page_index, page_text)

    def flush_prose():
        if prose_run:
            chunks.extend(_fixed_size_chunks(list(prose_run), chunk_size, overlap))
            prose_run.clear()

    i = 0
    while i < len(pages):
        if i < front_matter_cutoff and is_structural_page(pages[i]):
            flush_prose()
            run_start = i
            run_text = ""
            while i < len(pages) and i < front_matter_cutoff and is_structural_page(pages[i]):
                run_text += pages[i] + "\n"
                i += 1
            for j in range(0, len(run_text), MAX_STRUCTURAL_CHUNK_CHARS):
                piece = run_text[j : j + MAX_STRUCTURAL_CHUNK_CHARS].strip()
                if piece:
                    chunks.append({"text": piece, "page_index": run_start, "chunk_type": "structural"})
        else:
            prose_run.append((i, pages[i]))
            i += 1
    flush_prose()

    return chunks


def main():
    """Smoke test: python chunk_text.py FILE [FILE ...]"""
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print("usage: python chunk_text.py FILE [FILE ...]", file=sys.stderr)
        return 1

    for path in paths:
        pages = extractor_for(path)(path)
        chunks = chunk_pages(pages)
        print(f"=== {path.name} ===")
        print(f"  {len(pages)} pages/sections -> {len(chunks)} chunks")
        mid = chunks[len(chunks) // 2]
        print(f"  --- sample chunk (from page/section {mid['page_index']}) ---")
        print(" ", mid["text"][:500].replace("\n", "\n  "))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
