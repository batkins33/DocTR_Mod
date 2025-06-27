# Doctr OCR to CSV

This project provides a pipeline for converting scanned truck ticket PDFs into structured CSV reports.
It uses the **Doctr** OCR engine alongside YAML/CSV rules to detect vendors, ticket numbers,
manifest numbers and other key fields.

## Prerequisites

- **Python** 3.8 or newer (3.10+ recommended)
- **System tools**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  and [Poppler](http://blog.alivate.com.au/poppler-windows/) for PDF rendering
- Install the required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## Basic Usage

1. Place your PDFs in a directory or specify a single file.
2. Create these configuration files in the project folder:
    - `ocr_keywords.csv` – vendor keywords
    - `extraction_rules.yaml` – field extraction rules
    - `config.yaml` – runtime options (see `docs/sample_config.yaml` for an example)
3. Run the main script:
   ```bash
   python doctr_ocr_to_csv.py
   ```
   You will be prompted to choose a file or directory if not provided in `config.yaml`.

The pipeline converts each page to images, runs Doctr OCR, applies regex/ROI
rules to extract fields and writes CSV reports under `output/`.
`combined_ticket_numbers.csv` now contains one row for each processed page with
a `duplicate_ticket` flag so missing or repeated numbers are easy to spot. It
also creates exception CSVs:
`ticket_number_exceptions.csv` for pages with no ticket number and
`duplicate_ticket_exceptions.csv` for pages where the same vendor and ticket number combination occurs more than once and for pages that produced no OCR text.

## Documentation

- [User Guide](docs/USER_GUIDE.md) – step-by-step instructions and configuration examples
- [Developer Guide](docs/DEVELOPER_GUIDE.md) – architecture and extension points

