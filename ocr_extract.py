#!/usr/bin/env python3
import io

import fitz
import pytesseract
from PIL import Image


def extract_pdf_ocr(path, dpi=200):
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages.append(pytesseract.image_to_string(img))
    doc.close()
    return pages
