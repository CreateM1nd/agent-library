#!/usr/bin/env python3
import json
import time
import traceback
from pathlib import Path

from extract_text import LIBRARY_DIR
from store_book import store_book

CHECKPOINT_FILE = Path(__file__).parent / "library-checkpoint.json"
FAILURES_FILE = Path(__file__).parent / "library-failures.jsonl"
NO_TEXT_FILE = Path(__file__).parent / "library-no-text.jsonl"


def load_completed():
    if not CHECKPOINT_FILE.exists():
        return set()
    return set(json.loads(CHECKPOINT_FILE.read_text())["completed"])


def save_completed(completed):
    CHECKPOINT_FILE.write_text(json.dumps({
        "completed": sorted(completed),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))


def discover_files():
    return sorted(
        p for p in LIBRARY_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".pdf", ".epub")
    )


def main():
    files = discover_files()
    completed = load_completed()
    pending = [f for f in files if f.name not in completed]

    print(f"{len(files)} total files, {len(completed)} already done, {len(pending)} pending")

    with FAILURES_FILE.open("a") as fail_f, NO_TEXT_FILE.open("a") as notext_f:
        for i, path in enumerate(pending, start=1):
            print(f"[{i}/{len(pending)}] {path.name}")
            started = time.time()
            try:
                result = store_book(path)
            except Exception as e:
                elapsed = round(time.time() - started, 1)
                print(f"  FAILED after {elapsed}s: {e}")
                fail_f.write(json.dumps({
                    "file": path.name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }) + "\n")
                fail_f.flush()
                continue

            if result["total_chunks"] == 0:
                print(f"  WARNING: no extractable text (likely scanned/image-only PDF)")
                notext_f.write(json.dumps({"file": path.name}) + "\n")
                notext_f.flush()

            completed.add(path.name)
            save_completed(completed)

    print(f"\ndone: {len(completed)}/{len(files)} files stored")


if __name__ == "__main__":
    main()
