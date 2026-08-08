import json
import time

from extract_text import LIBRARY_DIR
from ocr_extract import extract_pdf_ocr
from store_book import store_book

MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

files = [json.loads(line)["file"] for line in open("library-no-text.jsonl")]

still_failing = []
for name in files:
    path = LIBRARY_DIR / name
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            store_book(path, extractor=extract_pdf_ocr)
            break
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                print(f"  STILL FAILING: {name}: {e}")
                still_failing.append(name)
            else:
                print(f"  attempt {attempt} failed for {name}: {e}, retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
    time.sleep(1)

print(f"\n{len(files) - len(still_failing)}/{len(files)} succeeded via OCR")
if still_failing:
    print("still failing:", still_failing)
