import os
import re
import csv
import yaml
import time
import hashlib
import logging
import numpy as np
from PIL import Image
from pathlib import Path

import cv2
import pytesseract
from doctr.models import ocr_predictor

# --- Config & Logging ---


def parse_roi(roi_cfg):
    return ((roi_cfg["x0"], roi_cfg["y0"]), (roi_cfg["x1"], roi_cfg["y1"]))


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- File Hash ---


def get_file_hash(filepath):
    return hashlib.sha256(filepath.encode("utf-8")).hexdigest()


# --- Image Extraction ---


def extract_images_generator(filepath, poppler_path=None):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        from pdf2image import convert_from_path

        for page in convert_from_path(filepath, dpi=300, poppler_path=poppler_path):
            yield page
    elif ext in [".tif", ".tiff"]:
        img = Image.open(filepath)
        while True:
            yield img.copy()
            try:
                img.seek(img.tell() + 1)
            except EOFError:
                break
    elif ext in [".jpg", ".jpeg", ".png"]:
        yield Image.open(filepath)
    else:
        raise ValueError("Unsupported file type")


# --- Orientation Correction ---


def correct_image_orientation(pil_img, page_num=None):
    try:
        osd = pytesseract.image_to_osd(pil_img)
        rotation_match = re.search(r"Rotate: (\d+)", osd)
        if rotation_match:
            rotation = int(rotation_match.group(1))
            logging.info(f"Page {page_num}: OSD rotation = {rotation} degrees")
            if rotation == 90:
                return pil_img.rotate(-90, expand=True)
            elif rotation == 180:
                return pil_img.rotate(180, expand=True)
            elif rotation == 270:
                return pil_img.rotate(-270, expand=True)
        # After orientation correction in serial processing

    except Exception as e:
        logging.warning(f"Orientation error (page {page_num}): {e}")
    return pil_img


# --- Ticket Extraction ---


def extract_ticket_number(result_page, roi):
    candidates = []
    for block in result_page.blocks:
        for line in block.lines:
            (lx_min, ly_min), (lx_max, ly_max) = line.geometry
            if (
                lx_min >= roi[0][0]
                and ly_min >= roi[0][1]
                and lx_max <= roi[1][0]
                and ly_max <= roi[1][1]
            ):
                text = " ".join(word.value for word in line.words)
                candidates.append(text)
    for text in candidates:
        m = re.search(r"\b\d{5,}\b", text)
        if m:
            return m.group(0)
    return candidates[0] if candidates else None


# --- Annotation/ROI ---


def draw_roi_only(page_image, roi, save_path=None, show=False):
    img = page_image.copy()
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape[:2]
    (x0, y0), (x1, y1) = roi
    pt1 = (int(x0 * width), int(y0 * height))
    pt2 = (int(x1 * width), int(y1 * height))
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)
    cv2.putText(
        img,
        "TICKET ROI",
        (pt1[0], max(0, pt1[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    if save_path:
        cv2.imwrite(save_path, img)
    if show:
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()


def draw_ocr_boxes(page_image, result_page, save_path=None, show=False):
    img = page_image.copy()
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape[:2]
    for block in result_page.blocks:
        for line in block.lines:
            (x_min, y_min), (x_max, y_max) = line.geometry
            pt1 = (int(x_min * width), int(y_min * height))
            pt2 = (int(x_max * width), int(y_max * height))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            text = " ".join(word.value for word in line.words)
            cv2.putText(
                img,
                text[:30],
                (pt1[0], max(0, pt1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 128, 255),
                1,
            )
    if save_path:
        cv2.imwrite(save_path, img)
    if show:
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()


# --- Per-Page Processing Function ---


def process_page(args):
    (page_idx, pil_img, cfg, file_hash, identifier) = args
    timings = {}
    page_num = page_idx + 1
    t0 = time.time()

    # Orientation
    if cfg.get("correct_orientation", True):
        pil_img = correct_image_orientation(pil_img, page_num=page_num)
    timings["orientation"] = time.time() - t0 if cfg.get("profile", False) else None

    # OCR
    t1 = time.time()
    img_np = np.array(pil_img)
    model = process_page.model  # ThreadPool shares this
    result = model([img_np])
    timings["ocr"] = time.time() - t1 if cfg.get("profile", False) else None

    # Ticket number
    t2 = time.time()
    ticket_number = extract_ticket_number(result.pages[0], cfg["roi"])
    timings["ticket"] = time.time() - t2 if cfg.get("profile", False) else None

    rows = []
    for block_idx, block in enumerate(result.pages[0].blocks):
        for line in block.lines:
            text = " ".join(word.value for word in line.words)
            position = line.geometry
            confidence = getattr(line, "confidence", 1.0)
            row = [
                identifier or "",  # identifier (optional, always included)
                file_hash,  # always present
                page_num,
                block_idx,
                "printed",
                text,
                position,
                confidence,
                ticket_number,
            ]
            rows.append(row)

    # (Optional) Save annotated images
    if cfg.get("save_images", False):
        output_dir = cfg["output_images_dir"]
        os.makedirs(output_dir, exist_ok=True)
        img_base = os.path.join(output_dir, f"marked_page_{page_num}")
        if cfg.get("draw_roi", False):
            draw_roi_only(
                img_np, cfg["roi"], save_path=img_base + "_roi.png", show=False
            )
        if cfg.get("draw_ocr_boxes", False):
            draw_ocr_boxes(
                img_np, result.pages[0], save_path=img_base + "_ocr.png", show=False
            )

    return rows, timings


# --- Main OCR to CSV Pipeline ---


def process_pdf_to_csv(cfg):
    file_hash = get_file_hash(cfg["input_pdf"])
    identifier = cfg.get("identifier", "")

    # Initialize OCR model once (shared in ThreadPool)
    model = ocr_predictor(pretrained=True)
    process_page.model = model

    corrected_images = [] if cfg.get("save_corrected_pdf", False) else None
    page_args = []

    # --- Collect all args and orientation-corrected images if needed ---
    for page_idx, pil_img in enumerate(
        extract_images_generator(cfg["input_pdf"], poppler_path=cfg.get("poppler_path"))
    ):
        if cfg.get("correct_orientation", True):
            pil_img = correct_image_orientation(pil_img, page_num=page_idx + 1)
        # Only store if needed
        if corrected_images is not None:
            corrected_images.append(pil_img.convert("RGB"))
        page_args.append((page_idx, pil_img, cfg, file_hash, identifier))

    results = []
    timings_total = []

    # --- Parallel or serial ---
    if cfg.get("parallel", False):
        from concurrent.futures import ThreadPoolExecutor

        logging.info(f"Running with {cfg.get('num_workers', 4)} parallel workers.")
        with ThreadPoolExecutor(max_workers=cfg.get("num_workers", 4)) as executor:
            for rows, timings in executor.map(process_page, page_args):
                results.extend(rows)
                timings_total.append(timings)
    else:
        logging.info("Running in serial mode.")
        for arg in page_args:
            rows, timings = process_page(arg)
            results.extend(rows)
            timings_total.append(timings)

    # After your CSV results are ready, extract unique (page, ticket_number) pairs
    ticket_numbers_per_page = []
    for row in results:
        page, ticket_number = row[2], row[-1]
        ticket_numbers_per_page.append((page, ticket_number))

    # --- Write ticket numbers by page ---
    # Only first ticket_number for each page is kept
    unique_ticket_numbers = {}
    for row in results:
        page, ticket_number = row[2], row[-1]
        if page not in unique_ticket_numbers:
            unique_ticket_numbers[page] = ticket_number
    ticket_numbers_csv = cfg.get("ticket_numbers_csv", "ticket_numbers.csv")
    os.makedirs(os.path.dirname(ticket_numbers_csv), exist_ok=True)  # <--- THIS LINE
    with open(ticket_numbers_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "ticket_number"])
        for page in sorted(unique_ticket_numbers):
            writer.writerow([page, unique_ticket_numbers[page]])
    logging.info(f"Ticket numbers by page saved to {ticket_numbers_csv}")

    # --- Write main OCR CSV ---
    output_csv = cfg["output_csv"]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "identifier",
                "file_hash",
                "page",
                "block_idx",
                "type",
                "text",
                "position",
                "confidence",
                "ticket_number",
            ]
        )
        for row in results:
            writer.writerow(row)

    # --- Save corrected PDF, only after all pages processed ---
    if corrected_images:
        corrected_pdf_path = cfg.get("corrected_pdf_path", "corrected_pages.pdf")
        try:
            corrected_images[0].save(
                corrected_pdf_path,
                save_all=True,
                append_images=corrected_images[1:],
                resolution=300,
            )
            logging.info(f"Rotated/corrected PDF saved as {corrected_pdf_path}")
        except Exception as e:
            logging.error(f"Could not save corrected PDF: {e}")

    # --- Profiling/timing ---
    if cfg.get("profile", False):
        import statistics

        for stage in ["orientation", "ocr", "ticket"]:
            stage_times = [t[stage] for t in timings_total if t[stage] is not None]
            if stage_times:
                logging.info(
                    f"{stage}: avg {statistics.mean(stage_times):.3f}s, "
                    f"max {max(stage_times):.3f}s, min {min(stage_times):.3f}s"
                )

    logging.info(f"OCR results saved to {cfg['output_csv']}")


# --- Entrypoint ---

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    cfg["roi"] = parse_roi(cfg["roi"])
    if not os.path.exists(cfg["input_pdf"]):
        logging.error(f"PDF file not found: {cfg['input_pdf']}")
        exit(1)
    if not os.path.exists(cfg["output_images_dir"]):
        os.makedirs(cfg["output_images_dir"])
    process_pdf_to_csv(cfg)
