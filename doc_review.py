import io
import sys
from pathlib import Path
from typing import Optional, TypedDict
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

# Constants
SUPPORTED_IMAGE_EXTS = {"jpg", "jpeg", "png"}
SUPPORTED_PDF_EXT = "pdf"
THRESHOLD_VALUE = 150
MAX_VALUE = 255
PDF_DPI = 300
OCR_LANG = "spa"
OCR_CONFIG = "--psm 6 -c preserve_interword_spaces=1"
NAME_KEYWORD = "NOMBRE"
MAX_PAGES_TO_PROCESS = 3  # Process first 3 pages for efficiency


class ExtractionResult(TypedDict):
    """Type definition for extraction result."""
    extracted_name: Optional[str]
    raw_text: str


def check_tesseract_installed() -> bool:
    """
    Check if Tesseract OCR is installed and accessible.
    
    Returns:
        True if Tesseract is available, False otherwise
    """
    try:
        pytesseract.get_tesseract_version()
        return True
    except (pytesseract.TesseractNotFoundError, Exception):
        return False


def ocr_image(image: Image.Image) -> str:
    """
    Perform OCR on a PIL Image with preprocessing.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text string
    """
    # Convert PIL → OpenCV
    img_array = np.array(image)
    
    # Convert to grayscale based on image mode
    if image.mode == 'L':
        # Already grayscale
        gray = img_array
    elif image.mode == 'RGBA':
        # RGBA: convert to BGR then grayscale
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif image.mode == 'RGB':
        # RGB: convert to grayscale directly
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        # Other modes: convert to RGB first
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding for better results
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)

    # Perform OCR
    try:
        text = pytesseract.image_to_string(binary, lang=OCR_LANG, config=OCR_CONFIG)
    except pytesseract.TesseractNotFoundError:
        error_msg = (
            "Tesseract OCR is not installed or not in your PATH.\n\n"
            "To install on macOS:\n"
            "  1. Install Homebrew (if not installed):\n"
            "     /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\n"
            "  2. Install Tesseract:\n"
            "     brew install tesseract tesseract-lang\n\n"
            "Alternatively, download from: https://github.com/tesseract-ocr/tesseract/wiki\n"
            "Make sure to add Tesseract to your PATH after installation."
        )
        raise RuntimeError(error_msg) from None
    return text


def extract_name(text: str) -> Optional[str]:
    """
    Extract name from OCR text by finding 'NOMBRE' keyword.
    
    Args:
        text: Full OCR text
        
    Returns:
        Extracted name string or None if not found
    """
    # Filter out empty lines and strip whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if NAME_KEYWORD in line.upper():
            # Try to extract name components from following lines
            try:
                apellido_paterno = lines[i + 1].strip()
                apellido_materno = lines[i + 2].strip()
                nombre = lines[i + 3].strip()
                
                # Validate that we got actual text (not empty)
                if all([apellido_paterno, apellido_materno, nombre]):
                    return f"{nombre} {apellido_paterno} {apellido_materno}"
            except (IndexError, AttributeError):
                continue

    return None


def get_name_from_file(file_bytes: bytes, filename: str, max_pages: int = MAX_PAGES_TO_PROCESS) -> ExtractionResult:
    """
    Extract name from image or PDF file.
    
    Args:
        file_bytes: File content as bytes
        filename: Original filename (used to determine file type)
        max_pages: Maximum number of PDF pages to process (default: 3)
        
    Returns:
        Dictionary with 'extracted_name' and 'raw_text' keys
        
    Raises:
        ValueError: If file type is not supported
    """
    # Get file extension
    ext = Path(filename).suffix.lower().lstrip(".")
    
    if ext in SUPPORTED_IMAGE_EXTS:
        # Process single image
        try:
            image = Image.open(io.BytesIO(file_bytes))
            full_text = ocr_image(image)
        except Exception as e:
            raise ValueError(f"Failed to process image: {e}") from e
            
    elif ext == SUPPORTED_PDF_EXT:
        # Process PDF (limit pages for efficiency)
        try:
            images = convert_from_bytes(file_bytes, dpi=PDF_DPI)
            # Process only first N pages for efficiency
            pages_to_process = min(len(images), max_pages)
            
            # Use list comprehension and join for better performance
            text_parts = [ocr_image(img) for img in images[:pages_to_process]]
            full_text = "\n".join(text_parts)
            
            # If name not found in first pages and we have more pages, process remaining
            name = extract_name(full_text)
            if name is None and len(images) > pages_to_process:
                remaining_text = "\n".join(
                    ocr_image(img) for img in images[pages_to_process:]
                )
                full_text += "\n" + remaining_text
            
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {e}") from e
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_IMAGE_EXTS)}, {SUPPORTED_PDF_EXT}")

    # Extract name from full text (only once, at the end)
    name = extract_name(full_text)

    return {
        "extracted_name": name,
        "raw_text": full_text
    }


def main():
    """Main function to test the document review system."""
    # Check if Tesseract is installed before processing
    if not check_tesseract_installed():
        print("ERROR: Tesseract OCR is not installed or not in your PATH.\n")
        print("To install on macOS:")
        print("  1. Install Homebrew (if not installed):")
        print('     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        print("  2. Install Tesseract:")
        print("     brew install tesseract tesseract-lang")
        print("\nAlternatively, download from: https://github.com/tesseract-ocr/tesseract/wiki")
        print("Make sure to add Tesseract to your PATH after installation.")
        sys.exit(1)
    
    file_path = Path("Vertical ID.jpeg")
    
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        return
    
    try:
        file_bytes = file_path.read_bytes()
        result = get_name_from_file(file_bytes, file_path.name)

        print("\n=== EXTRACTED NAME ===")
        print(result["extracted_name"] or "Name not found")

        print("\n=== RAW TEXT (first 800 chars) ===")
        print(result["raw_text"][:800])
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
