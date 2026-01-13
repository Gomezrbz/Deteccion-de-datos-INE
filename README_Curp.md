# CURP Second Version Name Extraction

This script extracts person names from CURP (Clave Única de Registro de Población) second version documents using OCR and pattern matching.

## Overview

The CURP document has a specific format where the name appears in a predictable pattern:
- A line containing numbers (typically a barcode number)
- The person's name on the next line
- A line starting with "PRESENTE" on the following line

**Example pattern:**
```
109016199400742
RAUL GOMEZ BERMUDEZ
PRESENTE Ciudad de México, a 20 de noviembre de 2024
```

The script uses Tesseract OCR to extract text from the image and then applies pattern matching to locate and extract the name.

## Requirements

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:
- `opencv-python`
- `numpy`
- `pillow`
- `pytesseract`
- `PyMuPDF`

### System Dependencies

**Tesseract OCR** must be installed on your system:

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

## Usage

### Command Line

Run the script with an image path and document type:

**Windows:**
```powershell
python Curp_Second_Version.py "data_testing/PNG/CURP_PNG2.png" "CURP second version"
```

**Linux/macOS:**
```bash
python3 Curp_Second_Version.py "data_testing/PNG/CURP_PNG2.png" "CURP second version"
```

### As a Python Module

You can also import and use the function in your own code:

```python
from Curp_Second_Version import extract_name_from_image

# Extract name from image
name = extract_name_from_image("path/to/image.png", "CURP second version")
if name:
    print(f"Extracted name: {name}")
else:
    print("Name not found")
```

## Function Reference

### `extract_name_from_image(image_path: str, doc_type: str) -> Optional[str]`

Extracts the name from a CURP second version document image.

**Parameters:**
- `image_path` (str): Path to the image file (PNG, JPG, or JPEG)
- `doc_type` (str): Document type identifier. Use "CURP second version" or "curp second" to trigger the CURP second version extraction logic.

**Returns:**
- `Optional[str]`: The extracted and cleaned name, or `None` if not found or an error occurred.

**Raises:**
- `FileNotFoundError`: If the image file doesn't exist
- `ValueError`: If the file cannot be opened as an image or OCR fails
- `RuntimeError`: If Tesseract OCR is not installed

## How It Works

1. **Image Loading**: The script loads the image file using PIL (Pillow).

2. **OCR Processing**: Tesseract OCR is used to extract text from the image. The script uses multiple preprocessing techniques and OCR configurations to improve accuracy.

3. **Pattern Matching**: The extracted text is searched for the specific pattern:
   - Find a line containing at least 10 digits (the barcode number)
   - Verify the next line looks like a name (using validation heuristics)
   - Verify the following line starts with "PRESENTE"
   - Extract and clean the name from the middle line

4. **Name Cleaning**: The extracted name is cleaned to:
   - Remove accents and normalize to uppercase
   - Fix common OCR errors
   - Remove non-letter characters (except spaces)
   - Normalize whitespace

## Example

Given the test file `data_testing/PNG/CURP_PNG2.png`, the script should extract:

```
Extracted name: RAUL GOMEZ BERMUDEZ
```

## Error Handling

The script handles various error conditions:

- **Missing file**: Returns a clear error message if the image file doesn't exist
- **Invalid image**: Reports if the file cannot be opened as an image
- **Tesseract not installed**: Provides installation instructions
- **OCR failures**: Reports OCR processing errors
- **Name not found**: Returns `None` if the pattern is not found in the document

## Limitations

- The script is specifically designed for CURP second version documents with the numbers → name → PRESENTE pattern
- OCR accuracy depends on image quality
- The pattern matching requires the exact sequence: numbers, name, PRESENTE
- Works best with high-resolution, clear images

## Related Files

- `methods.py`: Contains the core OCR and extraction functions
- `data_testing/PNG/CURP_PNG2.png`: Test image file
- `data_testing/PNG/output.txt.txt`: Example OCR output showing the extraction pattern
