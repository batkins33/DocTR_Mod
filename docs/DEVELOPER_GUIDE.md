---

## **`DEVELOPER_GUIDE.md`**

```markdown
# Doctr OCR to CSV - Developer/Technical Documentation

## Architecture Overview

- **Input:** PDF or directory of PDFs
- **Image Extraction:** Converts each PDF page to PIL image
- **OCR:** Uses Doctr (torch) for page-level text detection
- **Field Extraction:** Per-vendor logic, defined in YAML/CSV
- **Validation:** Manifest and ticket numbers validated via regex and length logic
- **Output:** Writes multiple CSVs (detailed, deduped, summary, exceptions) and ROI-marked images

---

## Code Structure

- **doctr_ocr_to_csv.py**: Main batch processing script
- **input_picker.py**: User prompt for file or directory selection
- **ocr_keywords.csv**: Vendor keywords
- **extraction_rules.yaml**: Field extraction definitions
- **config.yaml**: Main app configuration

---

## Key Functions & Flow

### Image & OCR

- `extract_images_generator(filepath, poppler_path)`
    - Converts PDF, TIFF, JPEG, PNG pages to images (yields PIL.Image)
- `ocr_predictor(pretrained=True)`
    - Loads Doctr OCR model

### Vendor and Field Extraction

- `find_vendor(page_text, vendor_rules)`
    - Scans OCR text for vendor keywords/exclusions (case-insensitive)
- `extract_vendor_fields(result_page, vendor_name, extraction_rules, pil_img, cfg)`
    - Extracts all needed fields for a page/vendor
- `extract_field(result_page, field_rules, pil_img, cfg)`
    - Core logic for ROI, label, regex-based extraction

### Validation

- **Manifest**:
    ```python
    def get_manifest_validation_status(manifest_number):
        if not manifest_number:
            return "invalid"
        if re.fullmatch(r"14\d{6}", manifest_number):
            return "valid"
        elif len(manifest_number) >= 7:
            return "review"
        else:
            return "invalid"
    ```
- **Ticket**: Vendor-specific regex

### Output & Stats

- Outputs all data as CSV, including ROI images if requested
- Deduplicates tickets by vendor and ticket number to flag pages that reuse the same ticket across the dataset
- Stats: total files, pages, valid/review/invalid manifests, missing tickets

---

## Extending & Customizing

- **Add new vendor:**
    - Update `ocr_keywords.csv`
    - Add extraction logic in `extraction_rules.yaml`
- **Change or add fields:**
    - Add to YAML and (optionally) extend extraction logic for new patterns
- **Custom validation:**
    - Edit `get_manifest_validation_status()` and `get_ticket_validation_status()`
- **Integrate to DB:**
    - Replace or supplement CSV writing logic

---

## Testing & Debugging

- **DEBUG mode:** Set `debug: true` in config.yaml for verbose output
- **Check ROIs:** Enable `draw_roi: true` to visually inspect extraction regions
- **Log output:** Adjust `logging.basicConfig()` for more/less detail

---

## Common Issues

- **Wrong field mapping:** Check output CSV column ordering and source field indices.
- **Low OCR quality:** Use better scans or preprocess images.
- **Field not extracting:** Tweak ROI or regex.
- **Parallel processing issues:** Some errors may be masked; run with `parallel: false` to debug.

---

## Sample Extraction Rule

```yaml
Waste Mgmt:
  ticket_number:
    method: roi
    roi: [0.18, 0.22, 0.29, 0.27]
    regex: '\d{6,}'
  manifest_number:
    method: label_right
    label: 'Manifest'
    regex: '14\d{6,}'
