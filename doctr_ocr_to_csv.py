import csv
import glob
import hashlib
import logging
import os
import re
import time

import numpy as np
import pandas as pd
import pytesseract
import yaml
from PIL import Image
from doctr.models import ocr_predictor

from input_picker import resolve_input


# --- Config & Logging ---
def load_extraction_rules(path="extraction_rules.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_roi(roi_cfg):
    return ((roi_cfg["x0"], roi_cfg["y0"]), (roi_cfg["x1"], roi_cfg["y1"]))


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


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
            matches = [m.strip().lower() for m in str(matches_str).split(",") if m.strip()]
        excludes_str = row.get("vendor_excludes", "")
        if pd.isna(excludes_str):
            excludes = []
        else:
            excludes = [e.strip().lower() for e in str(excludes_str).split(",") if e.strip()]
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


def is_valid_manifest_number(num):
    return bool(re.fullmatch(r"14\d{6}", num or ""))


def get_manifest_validation_status(manifest_number):
    if manifest_number is None:
        return "invalid"
    return "valid" if re.fullmatch(r"14\d{6}", manifest_number) else "invalid"


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
            roi_box = pil_img.crop((x0, y0, x1, y1))

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

        # **Filter out label if provided**
        if label:
            candidates = [c for c in candidates if label not in c.lower()]

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

    # Compose full OCR text for the page (after OCR!)
    full_text = " ".join(
        " ".join(word.value for word in line.words)
        for block in result.pages[0].blocks
        for line in block.lines
    )
    vendor_name, vendor_type, matched_term = find_vendor(full_text, vendor_rules)
    # ...after vendor_name, vendor_type, matched_term = find_vendor(full_text, vendor_rules)
    vendor_rule = extraction_rules.get(vendor_name, extraction_rules.get("DEFAULT"))

    # Save Images & Draw ROI from extraction rules
    if cfg.get("save_images", False):
        output_dir = cfg.get("output_images_dir", "./output/images")
        os.makedirs(output_dir, exist_ok=True)
        pil_img.save(os.path.join(output_dir, f"page_{page_num}.png"))

        if cfg.get("draw_roi", False):
            import cv2

            arr = np.array(pil_img)
            # Get ROI for ticket_number for this vendor
            field_rules = vendor_rule.get("ticket_number", {})
            roi = field_rules.get("roi") or field_rules.get("box")
            if roi:
                if max(roi) <= 1:  # normalized
                    width, height = pil_img.size
                    roi = [
                        [int(roi[0] * width), int(roi[1] * height)],
                        [int(roi[2] * width), int(roi[3] * height)],
                    ]
                else:
                    roi = [[roi[0], roi[1]], [roi[2], roi[3]]]
                cv2.rectangle(
                    arr,
                    (roi[0][0], roi[0][1]),
                    (roi[1][0], roi[1][1]),
                    (255, 0, 0),
                    2,
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"page_{page_num}_roi.png"), arr[..., ::-1]
                )

    field_rules = vendor_rule.get(
        "ticket_number", {}
    )  # or whichever field you want ROI for

    # Determine ROI based on field rules
    roi = None
    if field_rules.get("method") in ["roi", "box"]:
        roi = field_rules.get("roi") or field_rules.get("box")
        # If normalized, multiply by image size as in your extract_field()
        if roi and max(roi) <= 1:
            width, height = pil_img.size
            roi_pixels = [
                int(roi[0] * width),
                int(roi[1] * height),
                int(roi[2] * width),
                int(roi[3] * height),
            ]
            roi = [[roi_pixels[0], roi_pixels[1]], [roi_pixels[2], roi_pixels[3]]]
        else:
            roi = [[roi[0], roi[1]], [roi[2], roi[3]]]  # if already pixels

    # Ticket number
    t2 = time.time()
    fields = extract_vendor_fields(
        result.pages[0], vendor_name, extraction_rules, pil_img, cfg
    )

    ticket_number = fields["ticket_number"]
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
                ticket_valid_status,  # instead of "valid"/"invalid"
                manifest_valid_status,  # instead of "valid"/"invalid"
            ]
            rows.append(row)

    return (
        rows,
        timings,
        pil_img.convert("RGB") if cfg.get("save_corrected_pdf", False) else None,
    )


# --- Main OCR to CSV Pipeline ---
def process_pdf_to_csv(cfg, vendor_rules, extraction_rules, return_rows=False):
    file_hash = get_file_hash(cfg["input_pdf"])
    identifier = cfg.get("identifier", "")

    # Initialize OCR model once (shared in ThreadPool)
    model = ocr_predictor(pretrained=True)
    process_page.model = model

    corrected_images = [] if cfg.get("save_corrected_pdf", False) else None
    page_args = []

    for page_idx, pil_img in enumerate(
        extract_images_generator(cfg["input_pdf"], poppler_path=cfg.get("poppler_path"))
    ):
        if cfg.get("correct_orientation", True):
            pil_img = correct_image_orientation(pil_img, page_num=page_idx + 1)
        if corrected_images is not None:
            corrected_images.append(pil_img.convert("RGB"))
        page_args.append(
            (page_idx, pil_img, cfg, file_hash, identifier, extraction_rules)
        )

    results = []
    timings_total = []

    # --- PROCESS ALL PAGES FIRST ---
    if cfg.get("parallel", False):
        from concurrent.futures import ThreadPoolExecutor

        logging.info(f"Running with {cfg.get('num_workers', 4)} parallel workers.")
        with ThreadPoolExecutor(max_workers=cfg.get("num_workers", 4)) as executor:
            for rows, timings, corrected_img in executor.map(process_page, page_args):
                results.extend(rows)
                timings_total.append(timings)
                if cfg.get("save_corrected_pdf", False) and corrected_img is not None:
                    corrected_images.append(corrected_img)

    else:
        logging.info("Running in serial mode.")
        for arg in page_args:
            rows, timings, corrected_img = process_page(arg)
            results.extend(rows)
            timings_total.append(timings)
            if cfg.get("save_corrected_pdf", False) and corrected_img is not None:
                corrected_images.append(corrected_img)

    if return_rows:
        # Add file_name and file_path columns to every row for combined output
        for row in results:
            row.insert(0, cfg.get("file_path", ""))  # file_path
            row.insert(0, cfg.get("file_name", ""))  # file_name
        return results, corrected_images
    else:
        # Write CSV output per file (legacy behavior, optional)
        pass  # ... your existing per-file CSV writing code if needed ...


# --- Entrypoint ---
def main():
    cfg = load_config("config.yaml")
    cfg["DEBUG"] = cfg.get("debug", False)  # Ensure always present
    vendor_rules = load_vendor_rules_from_csv("ocr_keywords.csv")
    cfg = resolve_input(cfg)
    # ...rest of your main logic...

    batch_mode = cfg.get("batch_mode", False)
    all_results = []
    all_corrected_images = []

    if batch_mode and cfg.get("input_dir"):
        pdf_files = glob.glob(
            os.path.join(cfg["input_dir"], "**", "*.pdf"), recursive=True
        )
    elif cfg.get("input_pdf"):
        pdf_files = [cfg["input_pdf"]]
    else:
        raise ValueError("No input files or directory specified!")

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file} ...")
        # Create per-file config
        file_cfg = cfg.copy()
        file_cfg["input_pdf"] = pdf_file
        file_cfg["file_name"] = os.path.basename(pdf_file)
        file_cfg["file_path"] = pdf_file

        results, corrected_images = process_pdf_to_csv(
            file_cfg, vendor_rules, extraction_rules, return_rows=True
        )
        all_results.extend(results)
        if corrected_images:
            all_corrected_images.extend(corrected_images)

    # --- Write combined OCR data dump ---
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    with open(cfg["output_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file_name",
                "file_path",
                "identifier",
                "file_hash",
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

    # --- Build combined ticket summary (one ticket per file+page) ---
    unique_tickets = {}
    for row in all_results:
        (
            file_name,
            file_path,
            identifier,
            file_hash,
            page,
            block_idx,
            typ,
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
            ticket_valid,
            manifest_valid,
        ) = row

        key = (file_name, page)
        if key not in unique_tickets and ticket_number:
            unique_tickets[key] = (
                file_hash,
                ticket_number,
                ticket_valid,
                manifest_number,
                manifest_valid,
                vendor_name,
                vendor_type,
                matched_term,
                file_path,
            )

    # --- Write combined ticket number CSV ---
    os.makedirs(
        os.path.dirname(cfg["ticket_numbers_csv"]), exist_ok=True
    )  # <-- must go BEFORE 'open'

    with open(cfg["ticket_numbers_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file_name",
                "page",
                "ticket_number",
                "ticket_valid",
                "manifest_number",
                "manifest_valid",
                "vendor_name",
                "vendor_type",
                "matched_term",
                "file_hash",
                "file_path",
            ]
        )
        for (file_name, page), (
            file_hash,
            ticket_number,
            ticket_valid,
            manifest_number,
            manifest_valid,
            vendor_name,
            vendor_type,
            matched_term,
            file_path,
        ) in sorted(unique_tickets.items()):
            writer.writerow(
                [
                    file_name,
                    page,
                    ticket_number,
                    ticket_valid,
                    manifest_number,
                    manifest_valid,
                    vendor_name,
                    vendor_type,
                    matched_term,
                    file_hash,
                    file_path,
                ]
            )

    # --- Save corrected PDF (if needed) ---
    if cfg.get("save_corrected_pdf", False) and all_corrected_images:
        os.makedirs(os.path.dirname(cfg["corrected_pdf_path"]), exist_ok=True)
        all_corrected_images[0].save(
            cfg["corrected_pdf_path"],
            save_all=True,
            append_images=all_corrected_images[1:],
            resolution=300,
        )

    # --- Summary stats ---
    # Count unique pages processed
    num_pages = len(set((row[0], row[4]) for row in all_results))  # (file_name, page)

    # Count ticket numbers found (non-empty, unique by (file_name, page, ticket_number))
    ticket_numbers = set()
    manifest_numbers = set()
    for row in all_results:
        file_name = row[0]
        page = row[4]
        ticket_number = row[9]
        manifest_number = row[10]
        if ticket_number:
            ticket_numbers.add((file_name, page, ticket_number))
        if manifest_number:
            manifest_numbers.add((file_name, page, manifest_number))

    print(f"--- Summary ---")
    print(f"Pages processed: {num_pages}")
    print(f"Ticket numbers found: {len(ticket_numbers)}")
    print(f"Manifest numbers found: {len(manifest_numbers)}")
    print(
        f"All done! Results saved to {cfg['output_csv']} and {cfg['ticket_numbers_csv']}"
    )


if __name__ == "__main__":
    main()
