#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import fitz  # pymupdf
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# Root of the corpus. Every stored source_path is recorded relative to this, so
# moving the corpus does not invalidate what is already in the database.
LIBRARY_DIR = Path(os.environ.get("LIBRARY_DIR", "./library")).expanduser()


def extract_pdf(path):
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def extract_epub(path):
    book = epub.read_epub(str(path))
    pages = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        if text:
            pages.append(text)
    return pages


def extractor_for(path):
    return extract_pdf if path.suffix.lower() == ".pdf" else extract_epub


def main():
    """Smoke test: python extract_text.py FILE [FILE ...]"""
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print("usage: python extract_text.py FILE [FILE ...]", file=sys.stderr)
        return 1

    for path in paths:
        print(f"=== {path.name} ===")
        if not path.exists():
            print("  MISSING FILE")
            continue
        pages = extractor_for(path)(path)
        total_chars = sum(len(p) for p in pages)
        print(f"  {len(pages)} pages/sections, {total_chars} total characters")
        print("  --- sample from page/section 3 ---")
        sample = pages[2] if len(pages) > 2 else pages[0]
        print(" ", sample[:600].replace("\n", "\n  "))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
