# Doctr OCR to CSV - User Guide

## Introduction

Doctr OCR to CSV is a Python tool for batch-processing scanned truck ticket PDFs.  
It extracts vendor, ticket, manifest, and other key fields using configurable YAML/CSV rules, and outputs user-friendly
CSV reports.

---

## Prerequisites

- **Python**: 3.8+ (recommended 3.10+)
- **System Tools
  **: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), [Poppler](http://blog.alivate.com.au/poppler-windows/) (
  Windows: add Poppler's `bin` to your PATH)
- **Python Packages**:

---

## Quick Start

1. Place your PDFs in a directory, or specify a file.
2. Prepare these config files in your project folder:

- `ocr_keywords.csv`
- `extraction_rules.yaml`
- `config.yaml`

3. Run:

- Answer the prompt for `[F]ile` or `[D]irectory`.

---

## Configuration Files

### `ocr_keywords.csv`

| vendor_name | vendor_type | vendor_match     | vendor_excludes |
|-------------|-------------|------------------|-----------------|
| Waste Mgmt  | landfill    | wm, waste        |                 |
| NTX         | landfill    | ntx, north texas |                 |

- **vendor_match**: Comma-separated keywords (case-insensitive)
- **vendor_excludes**: Comma-separated terms to avoid false matches

---

### `extraction_rules.yaml`

YAML defining per-vendor extraction logic for fields:

```yaml
Waste Mgmt:
ticket_number:
 method: roi
 roi: [0.2, 0.7, 0.4, 0.8]
 regex: '\d{5,}'
manifest_number:
 method: label_right
 label: 'Manifest'
 regex: '14\d{6,}'
# ...other fields...
DEFAULT:
ticket_number: {...}
manifest_number: {...}


### `config.yaml`

input_pdf: ./data/sample.pdf
input_dir: ./data/
batch_mode: true
output_csv: ./output/ocr/all_results.csv
ticket_numbers_csv: ./output/ocr/ticket_numbers.csv
output_images_dir: ./output/images/
draw_roi: true
orientation_check: tesseract  # tesseract, doctr, or none
save_corrected_pdf: true
corrected_pdf_path: ./output/ocr/corrected.pdf
parallel: true
num_workers: 4
debug: false
profile: false

`orientation_check` determines how page rotation is handled:
- `tesseract` (default): use Tesseract's OSD to correct orientation
- `doctr`: use Doctr's angle prediction model
- `none`: skip orientation checks
