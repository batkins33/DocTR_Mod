import logging
import os

import cv2
import numpy as np
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path, pdfinfo_from_path
from tqdm import tqdm


def count_total_pages(pdf_files, cfg):
    """
    Return the total number of pages across all pdf_files.
    """
    total = 0
    for p in pdf_files:
        info = pdfinfo_from_path(p, poppler_path=cfg.get("poppler_path"))
        total += info["Pages"]
    return total


def is_page_ocrable(pdf_path, page_no, cfg):
    """
    Render page_no (1-based) at cfg["preflight"]["dpi_threshold"],
    convert to grayscale, Otsu-binarize a center crop, and see if
    we get at least cfg["preflight"]["min_chars"] from Tesseract.
    """
    dpi = cfg["preflight"].get("dpi_threshold", 150)
    min_chars = cfg["preflight"].get("min_chars", 5)
    poppler = cfg.get("poppler_path")

    # 1) Rasterize just that page
    imgs = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_no,
        last_page=page_no,
        poppler_path=poppler,
    )
    if not imgs:
        return False
    img = imgs[0]

    # 2) Crop & convert to L
    w, h = img.size
    crop = img.convert("L").crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    arr = np.array(crop)

    # 3) Otsu threshold to boost contrast
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    test_img = Image.fromarray(th)

    # 4) Quick OCR
    txt = pytesseract.image_to_string(test_img, config="--psm 6").strip()
    return len(txt) >= min_chars


def preflight_pages(pdf_path, cfg):
    """
    Return a list of (page_no, reason) for pages that fail OCRability.
    """
    bad = []
    total = count_total_pages([pdf_path], cfg)
    for p in tqdm(range(1, total + 1), desc="Preflight", unit="page"):
        if not is_page_ocrable(pdf_path, p, cfg):
            bad.append((p, "preflight-OCR failed"))
    return bad


def extract_page(pdf_path, page_no, out_path):
    """
    Write just that one page into a new single-page PDF at out_path.
    """
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_no - 1])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        writer.write(f)


def run_preflight(pdf_path, cfg):
    """
    If cfg["preflight"]["enabled"], run the OCRability check on each page,
    dump any bad pages out as single-page PDFs under cfg["exceptions_dir"],
    and return (skip_pages_set, exception_dicts_list).
    """
    skip_pages = set()
    exceptions = []

    pf = cfg.get("preflight", {})
    if not pf.get("enabled", False):
        return skip_pages, exceptions

    bad = preflight_pages(pdf_path, cfg)
    exc_dir = cfg.get("preflight_exceptions_dir", "./output/exceptions/preflight")

    for page, reason in bad:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        out_pdf = os.path.join(exc_dir, f"{stem}_page{page:03d}_preflight.pdf")
        extract_page(pdf_path, page, out_pdf)

        logging.warning(f"Preflight: page {page} => {reason}, saved to {out_pdf}")

        # Optional: also save a PNG image of the failed page
        try:
            dpi = pf.get("dpi_threshold", 150)
            poppler = cfg.get("poppler_path")
            imgs = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page,
                last_page=page,
                poppler_path=poppler,
            )
            if imgs:
                img = imgs[0]
                err_img_dir = cfg.get(
                    "preflight_error_images", "./output/exceptions/preflight_images"
                )
                os.makedirs(err_img_dir, exist_ok=True)
                out_img_path = os.path.join(err_img_dir, f"{stem}_page{page:03d}.png")
                img.save(out_img_path)
        except Exception as e:
            logging.warning(f"Could not save preflight image for page {page}: {e}")

        skip_pages.add(page)
        exceptions.append(
            {
                "file": pdf_path,
                "page": page,
                "error": reason,
                "extract": out_pdf,
            }
        )

    return skip_pages, exceptions
