#!/usr/bin/env python3
"""
Script to extract names from CURP second version documents.

This script processes CURP second version images and extracts the person's name
using OCR and pattern matching. The name is found using the pattern:
numbers → name → PRESENTE
"""

import sys
from pathlib import Path
from typing import Optional
from PIL import Image

from utils import (
    check_tesseract_installed,
    ocr_image,
    extract_name_curp_second_version,
)


def extract_name_from_image(image_path: str, doc_type: str) -> Optional[str]:
    """
    Extract name from an image file based on document type.
    
    Args:
        image_path: Path to the image file (PNG, JPG, JPEG)
        doc_type: Type of document (e.g., "CURP")
        
    Returns:
        Extracted name string, or None if not found or error occurred
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the file cannot be opened as an image
    """
    # Validate file exists
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    try:
        image = Image.open(path)
    except Exception as e:
        raise ValueError(f"Failed to open image file: {e}") from e
    
    # Perform OCR (use fast_mode for CURP documents which are usually upright)
    try:
        full_text = ocr_image(image, fast_mode=True)
    except RuntimeError as e:
        # Tesseract not installed error
        raise RuntimeError(str(e)) from e
    except Exception as e:
        raise ValueError(f"OCR failed: {e}") from e
    
    # Route to appropriate extraction function based on doc_type
    doc_type_lower = doc_type.lower()
    if doc_type_lower == "curp":
        return extract_name_curp_second_version(full_text)
    else:
        # For other document types, return None (could be extended later)
        return None


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python Curp_Second_Version.py <image_path> <doc_type>")
        print("\nExample:")
        print('  python Curp_Second_Version.py "data_testing/PNG/CURP_PNG2.png" "CURP"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    doc_type = sys.argv[2]
    
    # Check Tesseract installation
    if not check_tesseract_installed():
        print("ERROR: Tesseract OCR is not installed or not in PATH.", file=sys.stderr)
        print("\nTo install Tesseract:", file=sys.stderr)
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki", file=sys.stderr)
        print("  Linux: sudo apt-get install -y tesseract-ocr tesseract-ocr-spa", file=sys.stderr)
        print("  macOS: brew install tesseract tesseract-lang", file=sys.stderr)
        sys.exit(1)
    
    # Extract name
    try:
        name = extract_name_from_image(image_path, doc_type)
        if name:
            print(f"Extracted name: {name}")
        else:
            print("Name not found in the document.")
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
