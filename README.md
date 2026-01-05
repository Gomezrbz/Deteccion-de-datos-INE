# Deteccion-de-datos-INE

Small Python script to **extract the person's full name** from common Mexican documents and suggest a clean filename:

- **INE** (Credencial para votar)
- **Passport** (Pasaporte; supports MRZ parsing)
- **CURP** (Constancia de la CURP)

## Install

Python deps:

```bash
python3 -m pip install -r requirements.txt
```

System deps:

- **Tesseract** (required for OCR)
  - Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr tesseract-ocr-spa`
  - macOS: `brew install tesseract tesseract-lang`
- **Poppler** (required only for PDFs via `pdf2image`)
  - Ubuntu/Debian: `sudo apt-get install -y poppler-utils`
  - macOS: `brew install poppler`

## Usage

Set your input file path inside `doc_review.py` (variable `DEFAULT_INPUT_PATH`), then run:

```bash
python3 doc_review.py
```

Optional: override the path via CLI:

```bash
python3 doc_review.py "path/to/document.jpg"
```

Include debug output (doc type, extracted name, OCR text):

```bash
python3 doc_review.py "path/to/document.jpg" --print-text
```
