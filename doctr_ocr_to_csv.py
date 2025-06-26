"""
Batch OCR ticket processing pipeline for scanned PDF files.
- Extracts vendor, ticket, manifest, and other fields using YAML-based rules.
- Outputs page-level and deduped ticket CSVs, exception CSVs, and (optionally) corrected PDFs.
- Uses `ocr_keywords.csv` and `extraction_rules.yaml` for config.
"""

import csv
import glob
import hashlib
import io
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
import yaml
from PIL import Image
from doctr.models import ocr_predictor
from tqdm import tqdm

from input_picker import resolve_input
from preflight import run_preflight  # your new module

# To collect performance data
performance_data = []
start_time = time.time()


def ocr_with_fallback(pil_img, model):
    """
    Takes a PIL image + the doctr OCR model.
    Returns whatever model([img_np]) returns,
    but with a fallback pass in grayscale if
    the first pass produces zero blocks.
    """
    img_np = np.array(pil_img)

    # 1) First pass (color)
    docs = model([img_np])  # KEEP the container
    # “docs.pages” is your list of pages; check page 1’s blocks:
    if any(block.lines for block in docs.pages[0].blocks):
        return docs

    # 2) Fallback: grayscale + Otsu
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray3 = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    logging.info("Fallback OCR: using grayscale+Otsu")
    docs2 = model([gray3])
    return docs2


# --- Config & Logging ---
def load_extraction_rules(path="extraction_rules.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_roi(roi_cfg):
    return ((roi_cfg["x0"], roi_cfg["y0"]), (roi_cfg["x1"], roi_cfg["y1"]))


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_total_pages(pdf_files, cfg):
    from pdf2image import pdfinfo_from_path

    total_pages = 0
    for pdf_file in pdf_files:
        info = pdfinfo_from_path(pdf_file, poppler_path=cfg.get("poppler_path"))
        total_pages += info["Pages"]
    return total_pages


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# --- Vendor Matching: Use only CSV functions ---
def load_vendor_rules_from_csv(path):
    df = pd.read_csv(path)
    vendor_rules = []
    for _, row in df.iterrows():
        name = str(row["vendor_name"]).strip()
        vtype = str(row["vendor_type"]).strip() if "vendor_type" in row else ""
        # Robust multi-term parsing
        matches_str = row.get("vendor_match", "")
        if pd.isna(matches_str):
            matches = []
        else:
            matches = [
                m.strip().lower() for m in str(matches_str).split(",") if m.strip()
            ]
        excludes_str = row.get("vendor_excludes", "")
        if pd.isna(excludes_str):
            excludes = []
        else:
            excludes = [
                e.strip().lower() for e in str(excludes_str).split(",") if e.strip()
            ]
        vendor_rules.append(
            {
                "vendor_name": name,
                "vendor_type": vtype,
                "match_terms": matches,
                "exclude_terms": excludes,
            }
        )
    return vendor_rules


vendor_rules = load_vendor_rules_from_csv("ocr_keywords.csv")
extraction_rules = load_extraction_rules("extraction_rules.yaml")


def find_vendor(page_text, vendor_rules):
    page_text_lower = page_text.lower()
    for rule in vendor_rules:
        matched_terms = [
            term for term in rule["match_terms"] if term in page_text_lower
        ]
        found_exclude = any(
            exclude in page_text_lower for exclude in rule["exclude_terms"]
        )
        if matched_terms and not found_exclude:
            return rule["vendor_name"], rule["vendor_type"], matched_terms[0]  # <-- NEW
    return "", "", ""


# --- File Hash ---
def get_file_hash(filepath):
    return hashlib.sha256(filepath.encode("utf-8")).hexdigest()


# --- Page Hash ---
def get_image_hash(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


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
    except Exception as e:
        logging.warning(f"Orientation error (page {page_num}): {e}")
    return pil_img


# --- Ticket Extraction ---
def extract_vendor_fields(
    result_page, vendor_name, extraction_rules, pil_img=None, cfg=None
):
    vendor_rule = extraction_rules.get(vendor_name, extraction_rules.get("DEFAULT"))
    result = {}
    for field in [
        "ticket_number",
        "manifest_number",
        "material_type",
        "truck_number",
        "date",
    ]:
        field_rules = vendor_rule.get(field)
        if field_rules:
            result[field] = extract_field(result_page, field_rules, pil_img, cfg)
        else:
            result[field] = None

    return result


# Remove spaces/special characters for filenames
def safe_filename(val):
    return re.sub(r"[^\w\-]", "_", str(val))


def normalize_ticket_number(raw):
    if not raw:
        return raw
    raw = raw.strip()
    if raw.upper().startswith("NO"):
        # Remove 'NO' prefix and any space
        return re.sub(r"^NO\s*", "", raw, flags=re.IGNORECASE)
    elif re.match(r"^A\s?\d{5,6}$", raw, re.IGNORECASE):
        # Normalize 'A 038270' to 'A038270'
        return raw.replace(" ", "")
    return raw


def is_valid_manifest_number(num):
    return bool(re.fullmatch(r"14\d{6}", num or ""))


def get_manifest_validation_status(manifest_number):
    if not manifest_number:
        return "invalid"
    if re.fullmatch(r"14\d{6}", manifest_number):
        return "valid"
    elif len(manifest_number) >= 7:
        return "review"
    else:
        return "invalid"


def is_valid_ticket_number(ticket_number, validation_regex):
    if not ticket_number or not validation_regex:
        return False
    return bool(re.fullmatch(validation_regex, ticket_number))


def get_ticket_validation_status(ticket_number, validation_regex):
    if not validation_regex:
        return "not checked"
    if not ticket_number:
        return "invalid"
    return "valid" if re.fullmatch(validation_regex, ticket_number) else "invalid"


def extract_field(result_page, field_rules, pil_img=None, cfg=None):
    method = field_rules.get("method")
    regex = field_rules.get("regex")
    label = str(field_rules.get("label") or "").lower()
    DEBUG = cfg.get("DEBUG", False) if cfg else False

    if method in ["roi", "box"]:
        roi = field_rules.get("roi") or field_rules.get("box")
        if pil_img is not None and roi and len(roi) == 4 and max(roi) <= 1:
            width, height = pil_img.size
            x0, y0, x1, y1 = [
                int(roi[0] * width),
                int(roi[1] * height),
                int(roi[2] * width),
                int(roi[3] * height),
            ]

        candidates = []
        for block in result_page.blocks:
            for line in block.lines:
                (lx_min, ly_min), (lx_max, ly_max) = line.geometry
                if (
                    lx_min >= roi[0]
                    and ly_min >= roi[1]
                    and lx_max <= roi[2]
                    and ly_max <= roi[3]
                ):
                    text = " ".join(word.value for word in line.words)
                    candidates.append(text)

        # --- Multi-label support for label ignore ---
        labels = field_rules.get("label", "")
        if isinstance(labels, str):
            # split comma-separated strings into list, strip whitespace
            labels = [label_term.strip().lower() for label_term in labels.split(",") if label_term.strip()]
        elif isinstance(labels, list):
            labels = [label_term.lower() for label_term in labels]
        else:
            labels = []

        # **Filter out label if provided**
        labels = field_rules.get("label", "")
        if isinstance(labels, str):
            labels = [label_term.strip().lower() for label_term in labels.split(",") if label_term.strip()]
        elif isinstance(labels, list):
            labels = [label_term.lower() for label_term in labels]
        else:
            labels = []
        if labels:
            candidates = [
                c for c in candidates if not any(lbl in c.lower() for lbl in labels)
            ]

        if DEBUG:
            print("CANDIDATES:", candidates)

        for text in candidates:
            if regex:
                m = re.search(regex, text)
                if DEBUG:
                    print(
                        f"[DEBUG] Matching '{regex}' in '{text}' => {m.group(0) if m else None}"
                    )
                if m:
                    return m.group(0)

        return candidates[0] if candidates else None

    # --- BELOW LABEL method ---
    elif method == "below_label":
        label = field_rules.get("label", "").lower()
        lines = []
        for block in result_page.blocks:
            for line in block.lines:
                lines.append(" ".join(word.value for word in line.words))
        ticket_label_idx = None
        for i, line in enumerate(lines):
            if label in line.lower():
                ticket_label_idx = i
                break
        if ticket_label_idx is not None and ticket_label_idx + 1 < len(lines):
            target_line = lines[ticket_label_idx + 1]
            if regex:
                m = re.search(regex, target_line)
                if m:
                    return m.group(0)
            return target_line.strip()
        return None

    # --- LABEL RIGHT method ---
    elif method == "label_right":
        label = field_rules.get("label", "").lower()
        regex = field_rules.get("regex")
        for block in result_page.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                if label in line_text.lower():
                    # Try regex on right side of label
                    idx = line_text.lower().find(label)
                    after_label = line_text[idx + len(label) :]
                    if regex:
                        m = re.search(regex, after_label)
                        if m:
                            return m.group(0)
                    # Fallback: return everything after label
                    return after_label.strip()
        return None

    # --- Default: not found ---
    return None


# --- Per-Page Processing Function ---
def process_page(args):
    (page_idx, pil_img, cfg, file_hash, identifier, extraction_rules) = args
    timings = {}
    page_num = page_idx + 1
    page_image_hash = get_image_hash(pil_img)
    t0 = time.time()

    # Orientation
    if cfg.get("correct_orientation", True):
        pil_img = correct_image_orientation(pil_img, page_num=page_num)
        page_image_hash = get_image_hash(pil_img)
    timings["orientation"] = time.time() - t0 if cfg.get("profile", False) else None

    # OCR with color→grayscale fallback
    t1 = time.time()
    model = process_page.model
    result = ocr_with_fallback(pil_img, model)
    timings["ocr"] = time.time() - t1 if cfg.get("profile", False) else None

    # Compose full OCR text for the page (after OCR!)
    full_text = " ".join(
        " ".join(word.value for word in line.words)
        for block in result.pages[0].blocks
        for line in block.lines
    )
    vendor_name, vendor_type, matched_term = find_vendor(full_text, vendor_rules)
    vendor_rule = extraction_rules.get(vendor_name, extraction_rules.get("DEFAULT"))

    # --- PATCHED: Defer image saving until after ticket/vendor extracted ---

    t2 = time.time()
    fields = extract_vendor_fields(
        result.pages[0], vendor_name, extraction_rules, pil_img, cfg
    )
    ticket_number = normalize_ticket_number(fields["ticket_number"])
    vendor_str = safe_filename(vendor_name or "unknown")
    ticket_str = safe_filename(ticket_number or "none")
    base = f"{page_num:04d}_{vendor_str}_{ticket_str}"  # PATCHED HERE

    # Prepare output dir
    file_stem = os.path.splitext(os.path.basename(cfg["input_pdf"]))[0].replace(
        " ", "_"
    )

    # Define separate dirs
    base_image_dir = os.path.join(
        cfg.get("output_images_dir", "./output/images"), file_stem, "base"
    )
    roi_image_dir = os.path.join(
        cfg.get("output_images_dir", "./output/images"), file_stem, "roi"
    )
    os.makedirs(base_image_dir, exist_ok=True)
    os.makedirs(roi_image_dir, exist_ok=True)

    # Save base image
    pil_img.save(os.path.join(base_image_dir, f"{base}.png"))

    # Save ROI image if needed
    if cfg.get("draw_roi", False):
        arr = np.array(pil_img)
        field_rules = vendor_rule.get("ticket_number", {})
        roi = field_rules.get("roi") or field_rules.get("box")
        if roi and len(roi) == 4:
            try:
                if max(roi) <= 1:
                    width, height = pil_img.size
                    pt1 = (int(roi[0] * width), int(roi[1] * height))
                    pt2 = (int(roi[2] * width), int(roi[3] * height))
                else:
                    pt1 = (int(roi[0]), int(roi[1]))
                    pt2 = (int(roi[2]), int(roi[3]))
                cv2.rectangle(arr, pt1, pt2, (255, 0, 0), 2)
            except Exception as e:
                logging.warning(
                    f"ROI rectangle error on page {page_num}: {e} (roi={roi})"
                )
        else:
            logging.warning(
                f"ROI not defined or wrong length on page {page_num}: {roi}"
            )

        # PATCHED: Save ROI image using same naming format
        cv2.imwrite(os.path.join(roi_image_dir, f"{base}_roi.png"), arr[..., ::-1])

    manifest_number = fields["manifest_number"]
    material_type = fields["material_type"]
    truck_number = fields["truck_number"]
    date_extracted = fields["date"]
    timings["ticket"] = time.time() - t2 if cfg.get("profile", False) else None

    # Manifest validation (universal)
    manifest_valid_status = get_manifest_validation_status(manifest_number)

    # Ticket validation (vendor-specific)
    ticket_rule = vendor_rule.get("ticket_number", {})
    ticket_validation_regex = ticket_rule.get("validation_regex")
    ticket_valid_status = get_ticket_validation_status(
        ticket_number, ticket_validation_regex
    )

    rows = []
    for block_idx, block in enumerate(result.pages[0].blocks):
        for line in block.lines:
            text = " ".join(word.value for word in line.words)
            position = line.geometry
            confidence = getattr(line, "confidence", 1.0)
            row = [
                identifier or "",
                file_hash,
                page_image_hash,
                page_num,
                block_idx,
                "printed",
                text,
                position,
                confidence,
                ticket_number,
                manifest_number,
                material_type,
                truck_number,
                date_extracted,
                vendor_name,
                vendor_type,
                matched_term,
                ticket_valid_status,
                manifest_valid_status,
            ]

            rows.append(row)

    return (
        rows,
        timings,
        pil_img.convert("RGB") if cfg.get("save_corrected_pdf", False) else None,
    )


# Tell linters & IDEs “yes, this attribute exists”
process_page.model = None


# --- Main OCR to CSV Pipeline ---
def process_pdf_to_csv(cfg, vendor_rules, extraction_rules, return_rows=False):
    # ─── Setup ────────────────────────────────────────────────────────────────
    timing_steps = {}

    # STEP: Counting pages in PDF
    t0 = time.time()
    print("    - Counting pages in PDF...")
    file_hash = get_file_hash(cfg["input_pdf"])
    identifier = cfg.get("identifier", "")
    total_pages = count_total_pages([cfg["input_pdf"]], cfg)
    timing_steps["count_pages_sec"] = time.time() - t0

    # STEP: Running preflight checks (blank/low-DPI pages)
    t1 = time.time()
    print("    - Running preflight checks (blank/low-DPI pages)...")
    skip_pages, exceptions = run_preflight(cfg["input_pdf"], cfg)
    timing_steps["preflight_sec"] = time.time() - t1

    # STEP: Loading OCR model
    t2 = time.time()
    print("    - Loading OCR model...")
    model = ocr_predictor(pretrained=True)
    process_page.model = model
    timing_steps["load_model_sec"] = time.time() - t2

    # STEP: Collecting Results / Initialize Results
    t3 = time.time()
    print("    - Collecting Results...")
    corrected_images = [] if cfg.get("save_corrected_pdf", False) else None
    results = []
    timings_total = []
    processed_pages = 0
    timing_steps["init_results_sec"] = time.time() - t3

    # STEP: Loading Results / Images
    t4 = time.time()
    print("    - Loading Results...")
    all_images = list(
        extract_images_generator(cfg["input_pdf"], poppler_path=cfg.get("poppler_path"))
    )
    timing_steps["load_images_sec"] = time.time() - t4

    # STEP: Rotating Pages & Building page_args
    t5 = time.time()
    print("    - Rotating Pages...")
    page_args = []
    for idx, pil_img in enumerate(tqdm(all_images, desc="Preparing pages", unit="page")):
        page_num = idx + 1
        if page_num in skip_pages:
            logging.info(f"Skipping page {page_num} (preflight)")
            continue
        if cfg.get("correct_orientation", True):
            pil_img = correct_image_orientation(pil_img, page_num)
        if corrected_images is not None:
            corrected_images.append(pil_img.convert("RGB"))
        page_args.append((idx, pil_img, cfg, file_hash, identifier, extraction_rules))
    timing_steps["rotate_pages_sec"] = time.time() - t5

    total_to_process = len(page_args)

    # ─── Process Pages (parallel or serial with tqdm) ─────────────────────────
    if cfg.get("parallel", False):
        logging.info(f"Running with {cfg.get('num_workers',4)} parallel workers.")
        with ThreadPoolExecutor(max_workers=cfg.get("num_workers", 4)) as exe:
            futures = {exe.submit(process_page, arg): arg for arg in page_args}
            for fut in tqdm(
                as_completed(futures),
                total=total_to_process,
                desc="OCR pages",
                unit="page",
            ):
                arg = futures[fut]
                page_idx, pil_img = arg[0], arg[1]
                page_num = page_idx + 1

                try:
                    rows, timings, corrected_img = fut.result()
                except Exception as e:
                    err_dir = cfg.get("exceptions_dir", "./output/ocr/exceptions")
                    os.makedirs(err_dir, exist_ok=True)
                    fn = os.path.splitext(os.path.basename(cfg["input_pdf"]))[0]
                    err_path = os.path.join(
                        err_dir, f"{fn}_page{page_num:03d}_runtime.png"
                    )
                    pil_img.save(err_path)
                    logging.warning(f"Page {page_num} ERROR: {e!r} → {err_path}")
                    exceptions.append(
                        {
                            "file": cfg["input_pdf"],
                            "page": page_num,
                            "error": str(e),
                            "extract": err_path,
                        }
                    )
                    continue

                # on success
                processed_pages += 1
                results.extend(rows)
                timings_total.append(timings)
                if corrected_images is not None and corrected_img is not None:
                    corrected_images.append(corrected_img)

    else:
        logging.info("Running in serial mode.")
        for arg in tqdm(
            page_args, total=total_to_process, desc="OCR pages", unit="page"
        ):
            page_idx, pil_img = arg[0], arg[1]
            page_num = page_idx + 1

            try:
                rows, timings, corrected_img = process_page(arg)
            except Exception as e:
                err_dir = cfg.get("exceptions_dir", "./output/ocr/exceptions")
                os.makedirs(err_dir, exist_ok=True)
                fn = os.path.splitext(os.path.basename(cfg["input_pdf"]))[0]
                err_path = os.path.join(err_dir, f"{fn}_page{page_num:03d}_runtime.png")
                pil_img.save(err_path)
                logging.warning(f"Page {page_num} ERROR: {e!r} → {err_path}")
                exceptions.append(
                    {
                        "file": cfg["input_pdf"],
                        "page": page_num,
                        "error": str(e),
                        "extract": err_path,
                    }
                )
                continue

            processed_pages += 1
            results.extend(rows)
            timings_total.append(timings)
            if corrected_images is not None and corrected_img is not None:
                corrected_images.append(corrected_img)

    # ─── Return or Legacy CSV branch ─────────────────────────────────────────
    if return_rows:
        for row in results:
            row.insert(0, cfg.get("file_path", ""))
            row.insert(0, cfg.get("file_name", ""))
        return (
            results,
            corrected_images,
            processed_pages,
            total_pages,
            exceptions,
            timing_steps,
        )

    else:
        # … legacy per-file CSV code …
        pass


def build_roi_image_path(
    file_path,
    page_num,
    output_images_dir,
    output_csv,
    vendor_name="unknown",
    ticket_number="none",
):
    file_stem = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_")
    vendor_str = safe_filename(vendor_name or "unknown")
    ticket_str = safe_filename(ticket_number or "none")
    base = f"{int(page_num):04d}_{vendor_str}_{ticket_str}"
    image_output_dir = os.path.join(output_images_dir, file_stem)
    roi_image = os.path.join(image_output_dir, f"{base}_roi.png")
    roi_image_rel = os.path.relpath(roi_image, start=os.path.dirname(output_csv))
    return roi_image_rel


# --- Entrypoint ---
def main():
    cfg = load_config("config.yaml")
    cfg["DEBUG"] = cfg.get("debug", False)
    vendor_rules = load_vendor_rules_from_csv("ocr_keywords.csv")
    cfg = resolve_input(cfg)

    # ✅ Declare this early:
    all_exceptions = []

    batch_mode = cfg.get("batch_mode", False)
    all_results = []
    all_corrected_images = []
    sum_processed_pages = 0
    sum_total_pages = 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if batch_mode:
        summary_base = "batch"
    else:
        summary_base = Path(cfg["input_pdf"]).stem

    exceptions_csv = os.path.join(
        cfg.get("exceptions_csv_base", "./output/logs/exceptions"),
        f"exceptions_{summary_base}_{timestamp}.csv",
    )

    # --- gather list of PDFs ---
    if batch_mode and cfg.get("input_dir"):
        pdf_files = glob.glob(
            os.path.join(cfg["input_dir"], "**", "*.pdf"), recursive=True
        )
    elif cfg.get("input_pdf"):
        pdf_files = [cfg["input_pdf"]]
    else:
        raise ValueError("No input files or directory specified!")

    # --- per-file loop ---
    for i, pdf_file in enumerate(pdf_files, 1):
        print(
            f"\n--- [{i}/{len(pdf_files)}] Processing: {os.path.basename(pdf_file)} ---"
        )
        file_start = time.time()  # ⬅️ ADD THIS
        file_cfg = cfg.copy()
        file_cfg["input_pdf"] = pdf_file
        file_cfg["file_name"] = os.path.basename(pdf_file)
        file_cfg["file_path"] = pdf_file

        results, corrected_images, proc_pages, tot_pages, excs, timing_steps = (
            process_pdf_to_csv(
                file_cfg, vendor_rules, extraction_rules, return_rows=True
            )
        )

        # Capture per-file timing and performance stats
        tickets_found = sum(1 for row in results if row[11])
        file_duration = time.time() - file_start
        file_perf = {
            "file": os.path.basename(pdf_file),
            "pages": tot_pages,
            "tickets_found": tickets_found,
            "exceptions": len(excs),
            "duration_sec": round(file_duration, 2),
        }
        file_perf.update({k: round(v, 2) for k, v in timing_steps.items()})
        performance_data.append(file_perf)

        # collect results & images
        all_results.extend(results)
        if corrected_images:
            all_corrected_images.extend(corrected_images)

        # accumulate pages stats & exceptions
        sum_processed_pages += proc_pages
        sum_total_pages += tot_pages
        all_exceptions.extend(excs)

        print(f"Done: {os.path.basename(pdf_file)}")

    # --- Calculate stats before writing summary files ---
    unique_tickets = {}
    for row in all_results:
        vendor_name = row[16]
        ticket_number = row[11]
        dedupe_key = (vendor_name, ticket_number)
        if dedupe_key not in unique_tickets and ticket_number:
            # Build deduped summary of tickets. Each key is (vendor_name, ticket_number).
            # Value is a tuple matching the ticket_numbers_csv header order.
            unique_tickets[dedupe_key] = (
                row[5],  # page
                vendor_name,
                ticket_number,
                row[19],  # ticket_valid
                row[12],  # manifest_number
                row[20],  # manifest_valid
                row[17],  # vendor_type
                row[18],  # matched_term
                row[0],  # file_name
                row[1],  # file_path
                row[4],  # page_image_hash
                row[3],  # file_hash
            )

    # Count valid manifest numbers
    valid_manifests = set()
    for row in all_results:
        if row[20] == "valid" and row[12]:  # 20: manifest_valid, 12: manifest_number
            key = (row[0], row[5], row[12])  # (file_name, page, manifest_number)
            valid_manifests.add(key)
    valid_manifest_numbers = len(valid_manifests)

    # Count manifests for review
    review_manifests = set()
    for row in all_results:
        if row[20] == "review" and row[12]:  # 20: manifest_valid, 12: manifest_number
            key = (row[0], row[5], row[12])  # (file_name, page, manifest_number)
            review_manifests.add(key)
    review_manifest_numbers = len(review_manifests)

    # --- Track pages with at least one ticket number ---
    pages_with_ticket = set(
        (row[0], row[5]) for row in all_results if row[11]
    )  # (file_name, page)

    # --- All pages encountered ---
    all_pages = set((row[0], row[5]) for row in all_results)

    # --- Pages with NO ticket number ---
    no_ticket_pages_set = all_pages - pages_with_ticket

    # --- Build list for exception report ---
    no_ticket_pages = []
    for file_name, page in sorted(no_ticket_pages_set):
        # Optionally grab vendor name from the first matching row for this page
        row = next((r for r in all_results if r[0] == file_name and r[5] == page), None)
        vendor_name = row[16] if row else ""
        file_path = row[1] if row else ""
        no_ticket_pages.append(
            {
                "file_name": file_name,
                "file_path": file_path,
                "page": page,
                "vendor_name": vendor_name,
                "roi_image": build_roi_image_path(
                    file_path,
                    page,
                    cfg.get("output_images_dir", "./output/images"),
                    cfg["output_csv"],
                ),
            }
        )

    num_no_ticket_pages = len(no_ticket_pages)

    # --- Write combined OCR data dump ---
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    output_path = cfg["output_csv"]
    file_exists = os.path.isfile(output_path)
    all_results.sort(key=lambda row: (row[0], int(row[5])))  # (file_name, page_num)

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "file_name",
                    "file_path",
                    "identifier",
                    "file_hash",
                    "page_image_hash",
                    "page",
                    "block_idx",
                    "type",
                    "text",
                    "position",
                    "confidence",
                    "ticket_number",
                    "manifest_number",
                    "material_type",
                    "truck_number",
                    "date",
                    "vendor_name",
                    "vendor_type",
                    "matched_term",
                    "ticket_valid",
                    "manifest_valid",
                ]
            )
        for row in all_results:
            writer.writerow(row)

    # --- Write combined ticket numbers CSV ---
    os.makedirs(os.path.dirname(cfg["ticket_numbers_csv"]), exist_ok=True)
    ticket_csv_path = cfg["ticket_numbers_csv"]
    file_exists = os.path.isfile(ticket_csv_path)

    # Sort tickets by file_name (v[8]) and page (v[0]) before writing:
    sorted_unique_tickets = sorted(
        unique_tickets.values(), key=lambda v: (v[8], int(v[0]))
    )

    with open(ticket_csv_path, "a", newline="", encoding="utf-8") as tf:
        w = csv.writer(tf)
        if not file_exists:
            w.writerow(
                [
                    "page",
                    "vendor_name",
                    "ticket_number",
                    "ticket_valid",
                    "manifest_number",
                    "manifest_valid",
                    "vendor_type",
                    "matched_term",
                    "file_name",
                    "file_path",
                    "page_image_hash",
                    "file_hash",
                ]
            )
        for vals in unique_tickets.values():
            w.writerow(vals)

    # --- Write ROI exception report for pages with no ticket ---
    roi_ex_csv = cfg["ticket_number_exceptions_csv"]
    os.makedirs(os.path.dirname(roi_ex_csv), exist_ok=True)
    file_exists = os.path.isfile(roi_ex_csv)

    with open(roi_ex_csv, "a", newline="", encoding="utf-8") as ef:
        w = csv.writer(ef)
        if not file_exists:
            w.writerow(
                ["file_name", "file_path", "page", "vendor_name", "ROI Image Link"]
            )
        for rec in no_ticket_pages:
            link = f'=HYPERLINK("{rec["roi_image"]}","View ROI")'
            w.writerow(
                [
                    rec["file_name"],
                    rec["file_path"],
                    rec["page"],
                    rec["vendor_name"],
                    link,
                ]
            )

    # --- Write exception report ONLY if exceptions exist ---
    if all_exceptions:
        os.makedirs(os.path.dirname(exceptions_csv), exist_ok=True)
        with open(exceptions_csv, "w", newline="", encoding="utf-8") as ef:
            writer = csv.writer(ef)
            writer.writerow(["file", "page", "error", "extract"])
            for ex in all_exceptions:
                writer.writerow([ex["file"], ex["page"], ex["error"], ex["extract"]])
        logging.info(f"Wrote exception report to {exceptions_csv}")
    else:
        logging.info("No exceptions encountered — skipping exception report.")

    # --- Write updated summary report CSV ---
    summary_dir = cfg.get("summary_report_dir", "./output/logs")
    summary_csv = os.path.join(summary_dir, f"summary_{summary_base}_{timestamp}.csv")
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    with open(summary_csv, "w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(["statistic", "value"])
        writer.writerow(["Files processed", len(pdf_files)])
        writer.writerow(["Total pages", sum_total_pages])
        writer.writerow(["Pages processed", sum_processed_pages])
        writer.writerow(["Pages not processed", sum_total_pages - sum_processed_pages])
        writer.writerow(["OCR data fields", len(all_results)])
        writer.writerow(["Unique tickets", len(unique_tickets)])
        writer.writerow(["Valid manifest numbers", valid_manifest_numbers])
        writer.writerow(["Manifest numbers for review", review_manifest_numbers])
        writer.writerow(["Pages with no ticket number", num_no_ticket_pages])
    logging.info(f"Wrote summary report to {summary_csv}")

    # --- Save corrected PDF, only after all pages processed ---
    if cfg.get("save_corrected_pdf", False) and all_corrected_images:
        corrected_pdf_path = cfg.get(
            "corrected_pdf_path", "./output/final_pdf/corrected_pages.pdf"
        )
        try:
            os.makedirs(os.path.dirname(corrected_pdf_path), exist_ok=True)
            all_corrected_images[0].save(
                corrected_pdf_path,
                save_all=True,
                append_images=all_corrected_images[1:],
                resolution=300,
            )
            logging.info(f"Rotated/corrected PDF saved as {corrected_pdf_path}")
        except Exception as e:
            logging.error(f"Could not save corrected PDF: {e}")

    # Write performance CSV
    perf_csv = "./output/ocr/performance_report.csv"
    os.makedirs(os.path.dirname(perf_csv), exist_ok=True)
    file_exists = os.path.isfile(perf_csv)

    perf_fieldnames = [
        "file",
        "pages",
        "tickets_found",
        "exceptions",
        "duration_sec",
        "count_pages_sec",
        "preflight_sec",
        "load_model_sec",
        "init_results_sec",
        "load_images_sec",
        "rotate_pages_sec",
    ]

    with open(perf_csv, "a", newline="", encoding="utf-8") as pf:
        writer = csv.DictWriter(pf, fieldnames=perf_fieldnames)
        if not file_exists:
            writer.writeheader()  # <--- Write header if new file
        writer.writerows(performance_data)  # <--- Actually write the data rows!

    logging.info(f"Wrote performance report to {perf_csv}")

    # --- Console printout (matching CSV) ---
    print(f"--- Summary ---")
    print(f"Files processed:     {len(pdf_files)}")
    print(f"Total pages:         {sum_total_pages}")
    print(f"Pages processed:     {sum_processed_pages}")
    print(f"Pages not processed: {sum_total_pages - sum_processed_pages}")
    print(f"OCR data fields:     {len(all_results)}")
    print(f"Unique tickets:      {len(unique_tickets)}")
    print(f"Valid manifests:     {valid_manifest_numbers}")
    print(f"Manifests for review:{review_manifest_numbers}")
    print(f"No-ticket pages:      {num_no_ticket_pages}")
    print(
        f"All done! Results saved to {cfg['output_csv']} and {cfg['ticket_numbers_csv']}"
    )


elapsed = time.time() - start_time
logging.info(f"Total processing time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
