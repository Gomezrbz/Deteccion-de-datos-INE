# Deteccion-de-datos-INE

Small Python script to **extract the person's full name** from common Mexican documents and suggest a clean filename:

- **INE** (Credencial para votar)
- **Passport** (Pasaporte; supports MRZ parsing)
- **CURP** (Constancia de la CURP)

## Install

### Python Dependencies

**Windows:**
```powershell
python -m pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python3 -m pip install -r requirements.txt
```

### System Dependencies

#### Tesseract OCR (required for OCR)

**Windows:**
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and make sure to:
   - Install the Spanish language pack (`tesseract-ocr-spa`)
   - Add Tesseract to your system PATH during installation
3. Verify installation:
   ```powershell
   tesseract --version
   ```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

> **Note:** PDF processing is handled by PyMuPDF (a pure Python library), so no additional system dependencies like Poppler are required.

## Usage

### Quick Start

Set your input file path inside `methods.py` (variable `DEFAULT_INPUT_PATH`), then run:

**Windows:**
```powershell
python doc_review.py
```

**Linux/macOS:**
```bash
python3 doc_review.py
```

### Command Line Options

Override the path via CLI:

**Windows:**
```powershell
python doc_review.py "path\to\document.jpg"
```

**Linux/macOS:**
```bash
python3 doc_review.py "path/to/document.jpg"
```

Include debug output (doc type, extracted name, OCR text):

**Windows:**
```powershell
python doc_review.py "path\to\document.jpg" --print-text
```

**Linux/macOS:**
```bash
python3 doc_review.py "path/to/document.jpg" --print-text
```

### Testing on Different Platforms

The script works identically across Windows, Linux, and macOS. Make sure:
- Python 3.7+ is installed
- Tesseract OCR is installed and accessible in your PATH
- All Python dependencies are installed via `pip install -r requirements.txt`

> **Note:** PDF processing uses PyMuPDF (pure Python), so no system dependencies like Poppler are needed.
