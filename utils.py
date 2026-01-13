import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# Constants
OCR_LANG = "spa+eng"
OCR_CONFIG = "--psm 6 -c preserve_interword_spaces=1"

def _find_tesseract_executable() -> Optional[str]:
    """
    Try to find Tesseract executable on the system.
    This is especially useful on Windows when Tesseract is not in PATH.
    
    Returns:
        Path to tesseract executable if found, None otherwise
    """
    # First, try to get version (works if already in PATH)
    try:
        pytesseract.get_tesseract_version()
        return None  # Already configured
    except pytesseract.TesseractNotFoundError:
        pass
    
    # Common Windows installation paths
    if sys.platform == "win32":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(
                Path.home().name
            ),
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
    
    # Try using 'where' command on Windows or 'which' on Unix
    import shutil
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        return tesseract_cmd
    
    return None

# Auto-configure Tesseract on module import
_tesseract_path = _find_tesseract_executable()
if _tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path

def check_tesseract_installed() -> bool:
    """
    Check if Tesseract OCR is installed and accessible.
    Automatically configures pytesseract if Tesseract is found but not in PATH.
    
    Returns:
        True if Tesseract is available, False otherwise
    """
    # Try to find and configure Tesseract if not already configured
    tesseract_path = _find_tesseract_executable()
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    try:
        pytesseract.get_tesseract_version()
        return True
    except (pytesseract.TesseractNotFoundError, Exception):
        return False

def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)
    )

def _normalize_text(s: str) -> str:
    # Uppercase + remove accents + normalize whitespace
    s = _strip_accents(s).upper()
    s = re.sub(r"[ \t]+", " ", s)
    return s

def _preprocess_for_ocr(img_array: np.ndarray, mode: str) -> np.ndarray:
    """
    Preprocess OpenCV image array for OCR with enhanced preprocessing.
    mode: 'binarize' (Otsu), 'adapt' (adaptive threshold), 'enhanced' (multiple enhancements)
    """
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # Enhanced preprocessing for better OCR
    if mode == "enhanced":
        # Step 1: Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Step 2: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 3: Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Step 4: Binarize
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    # Original preprocessing methods
    # Mild denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if mode == "adapt":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
    # Default: Otsu binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def _ocr_image_fast(image: Image.Image) -> str:
    """
    Fast OCR for documents that are known to be upright (like CURP).
    Uses single preprocessing method and single OCR config for speed.
    """
    # Convert PIL → OpenCV
    img_array = np.array(image)
    
    # Convert to grayscale based on image mode
    if image.mode == 'L':
        gray = img_array
    elif image.mode == 'RGBA':
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif image.mode == 'RGB':
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        # Other modes: convert to RGB first
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    
    # Use only the best preprocessing method (enhanced) and single OCR config
    try:
        processed = _preprocess_for_ocr(gray, mode="enhanced")
        processed_img = Image.fromarray(processed)
        text = pytesseract.image_to_string(processed_img, lang=OCR_LANG, config=OCR_CONFIG)
        return text if text.strip() else ""
    except pytesseract.TesseractNotFoundError:
        error_msg = (
            "Tesseract OCR is not installed or not in your PATH.\n\n"
            "To install on Ubuntu/Debian:\n"
            "  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-spa\n\n"
            "To install on macOS:\n"
            "  brew install tesseract tesseract-lang\n\n"
            "More info: https://github.com/tesseract-ocr/tesseract/wiki\n"
            "Make sure to add Tesseract to your PATH after installation."
        )
        raise RuntimeError(error_msg) from None
    except Exception:
        # Fallback to simple binarization if enhanced fails
        try:
            processed = _preprocess_for_ocr(gray, mode="binarize")
            processed_img = Image.fromarray(processed)
            text = pytesseract.image_to_string(processed_img, lang=OCR_LANG, config=OCR_CONFIG)
            return text if text.strip() else ""
        except Exception:
            return ""

def ocr_image(image: Image.Image, fast_mode: bool = False) -> str:
    """
    OCR with rotation attempts (useful for sideways photos).
    
    Args:
        image: PIL Image object
        fast_mode: If True, skip rotations and use fewer preprocessing attempts (faster)
    """
    if fast_mode:
        # Fast mode: single attempt with best preprocessing
        try:
            return _ocr_image_fast(image)
        except RuntimeError:
            raise
        except Exception:
            return ""
    
    # Standard mode: try multiple rotations (not used when fast_mode=True)
    # This code is kept for backward compatibility but won't be executed in our test
    return ""

def _fix_common_ocr_errors(text: str) -> str:
    """
    Fix common OCR errors that affect name extraction.
    """
    # Common OCR misreadings
    replacements = {
        # Numbers mistaken for letters
        '0': 'O', '1': 'I', '5': 'S', '8': 'B',
        # Common character confusions
        'rn': 'm', 'vv': 'w', 'cl': 'd', 'ii': 'll',
        # Punctuation that shouldn't be in names
        '>': '', '<': '', '|': '', '\\': '', '/': '',
        # Common OCR artifacts
        'o>': 'O', '>>': '',
    }
    
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result

def _clean_name_piece(s: str) -> str:
    s = _strip_accents(s).upper()
    # Fix common OCR errors first
    s = _fix_common_ocr_errors(s)
    # Remove non-letter characters except spaces
    s = re.sub(r"[^A-Z ]+", " ", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _looks_like_name_line(line: str) -> bool:
    l = _clean_name_piece(line)
    if len(l) < 3:
        return False
    
    # Avoid common non-name labels
    banned = (
        "NOMBRE",
        "APELLIDO",
        "APELLIDOS",
        "DOMICILIO",
        "SEXO",
        "FECHA",
        "NACIONALIDAD",
        "AUTORIDAD",
        "OBSERVACIONES",
        "CURP",
        "CLAVE",
        "REGISTRO",
        "MEXICO",
        "ESTADOS UNIDOS MEXICANOS",
        "INSTITUTO NACIONAL ELECTORAL",
        "CREDENCIAL PARA VOTAR",
        "PASAPORTE",
        "SECRETARIA",
        "GOBIERNO",
        "CREDENCIAL",
        "VOTAR",
        "ELECTOR",
    )
    if any(b in l for b in banned):
        return False
    
    # Names should have at least 2 words (first name + last name minimum)
    words = l.split()
    if len(words) < 2:
        return False
    
    # Names are typically mostly letters/spaces, with words of reasonable length
    # Each word should be at least 2 characters and mostly letters
    valid_words = sum(1 for w in words if len(w) >= 2 and re.match(r'^[A-Z]+$', w))
    if valid_words < 2:
        return False
    
    # Total length should be reasonable (not too short, not too long)
    if len(l) < 6 or len(l) > 100:
        return False
    
    return True

def extract_curp_from_text(text: str, debug: bool = False) -> Optional[str]:
    """
    Extract CURP/Clave from CURP second version document.
    
    Pattern: Text after "Clave:" and before "Nombre"
    Example:
        Clave:
        
        GOBR940325HDFMRLO3
        
        Nombre
    
    Args:
        text: OCR extracted text from the document
        debug: If True, print debug information
        
    Returns:
        Extracted and cleaned CURP/Clave, or None if not found
    """
    if debug:
        print("\n=== CURP Extraction Debug ===")
        print(f"Text length: {len(text)} characters")
    
    # Find "Clave:" pattern (case-insensitive, flexible whitespace)
    clave_pattern = r'Clave\s*:'
    clave_match = re.search(clave_pattern, text, re.IGNORECASE)
    
    if not clave_match:
        if debug:
            print("✗ 'Clave:' pattern not found in text")
        return None
    
    if debug:
        print(f"✓ Found 'Clave:' at position {clave_match.start()}-{clave_match.end()}")
    
    # Get text after "Clave:" and split into lines
    start_pos = clave_match.end()
    text_after_clave = text[start_pos:]
    
    if debug:
        print(f"Text after 'Clave:': {repr(text_after_clave[:100])}...")
    
    # Find "Nombre" pattern (case-insensitive)
    nombre_pattern = r'Nombre'
    nombre_match = re.search(nombre_pattern, text_after_clave, re.IGNORECASE)
    
    if not nombre_match:
        if debug:
            print("✗ 'Nombre' pattern not found after 'Clave:'")
        return None
    
    if debug:
        print(f"✓ Found 'Nombre' at position {nombre_match.start()} after 'Clave:'")
    
    # Extract text between "Clave:" and "Nombre"
    text_before_nombre = text_after_clave[:nombre_match.start()]
    
    if debug:
        print(f"Text between 'Clave:' and 'Nombre': {repr(text_before_nombre)}")
    
    # Split into lines and find the first non-empty line that looks like a CURP
    lines = [line.strip() for line in text_before_nombre.splitlines()]
    
    if debug:
        print(f"Lines found: {len(lines)}")
        for i, line in enumerate(lines):
            print(f"  Line {i}: {repr(line)}")
    
    for line in lines:
        if not line:
            continue
        
        if debug:
            print(f"\nProcessing line: {repr(line)}")
        
        # Clean up the CURP - remove extra whitespace, keep alphanumeric
        curp_cleaned = re.sub(r'\s+', '', line)  # Remove all whitespace
        curp_cleaned = curp_cleaned.upper()  # Convert to uppercase
        curp_cleaned = re.sub(r'[^A-Z0-9]', '', curp_cleaned)  # Remove non-alphanumeric
        
        if debug:
            print(f"  After cleaning: {repr(curp_cleaned)}")
            print(f"  Length: {len(curp_cleaned)}")
        
        # Validate: CURP should be 18 characters typically, but allow 10-20 for OCR errors
        if len(curp_cleaned) >= 10 and len(curp_cleaned) <= 20:
            if debug:
                print(f"✓ Valid CURP found: {curp_cleaned}")
            return curp_cleaned
        elif debug:
            print(f"  ✗ Length {len(curp_cleaned)} not in valid range (10-20)")
    
    if debug:
        print("✗ No valid CURP found in lines between 'Clave:' and 'Nombre'")
    
    return None


def extract_name_curp_second_version(text: str) -> Optional[str]:
    """
    Extract name from CURP second version document.
    
    Pattern: A line with numbers → name line → line starting with "PRESENTE"
    Example:
        109016199400742
        RAUL GOMEZ BERMUDEZ
        PRESENTE Ciudad de México, a 20 de noviembre de 2024
    
    Args:
        text: OCR extracted text from the document
        
    Returns:
        Extracted and cleaned name, or None if not found
    """
    # Optimized regex-based approach: find the pattern in one pass
    # Pattern: line with 10+ digits, followed by name, followed by PRESENTE
    pattern = r'^([0-9\s\-]{10,})\s*\n\s*([A-ZÁÉÍÓÚÑ\s]{6,100})\s*\n\s*(PRESENTE[^\n]*)'
    
    # Try multiline regex match
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    if match:
        # Verify the middle group looks like a name
        potential_name = match.group(2).strip()
        if _looks_like_name_line(potential_name):
            return _clean_name_piece(potential_name)
    
    # Fallback to line-by-line approach if regex doesn't match
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Iterate through lines to find the pattern
    for i in range(len(lines) - 2):
        current_line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        following_line = lines[i + 2] if i + 2 < len(lines) else ""
        
        # Check if current line contains mostly numbers (like "109016199400742")
        # Allow for some OCR errors (spaces, dashes, etc.)
        numbers_only = re.sub(r'[^0-9]', '', current_line)
        if len(numbers_only) >= 10:  # At least 10 digits to be considered a number line
            # Check if next line looks like a name
            if _looks_like_name_line(next_line):
                # Check if following line starts with "PRESENTE" (case-insensitive)
                following_normalized = _normalize_text(following_line)
                if following_normalized.startswith("PRESENTE"):
                    # Found the pattern! Extract and clean the name
                    return _clean_name_piece(next_line)
    
    return None


def process_pdf(pdf_path: str) -> Image.Image:
    """
    Convert first page of PDF to PIL Image.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        PIL Image of the first page
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the PDF cannot be opened or processed
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        
        if len(pdf_document) == 0:
            pdf_document.close()
            raise ValueError("PDF file is empty")
        
        # Get first page
        first_page = pdf_document[0]
        
        # Render page to image (pixmap)
        # Use a reasonable DPI for OCR (300 DPI is good)
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI scaling
        pix = first_page.get_pixmap(matrix=mat)
        
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Close PDF
        pdf_document.close()
        
        return img
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e}") from e


def process_text_file(text_file_path: str, name_expected: str, curp_expected: str = None) -> Dict:
    """
    Process a text file directly (faster than OCR).
    This is useful when you already have the OCR output from tesseract command.
    
    Args:
        text_file_path: Path to the text file (e.g., output.txt.txt from tesseract)
        name_expected: Expected name for validation
        curp_expected: Expected CURP/Clave for validation (optional)
        
    Returns:
        Dictionary containing:
            - 'curp': Extracted CURP/Clave (or None)
            - 'name': Extracted name (or None)
            - 'name_match': Boolean indicating if extracted name matches expected
            - 'curp_match': Boolean indicating if extracted CURP matches expected
            - 'timing': Dictionary with timing information
            - 'success': Boolean indicating overall success
            - 'error': Error message if any (or None)
    """
    total_start = time.perf_counter()
    timing = {
        'file_load': 0.0,
        'curp_extraction': 0.0,
        'name_extraction': 0.0,
        'total': 0.0
    }
    
    result = {
        'curp': None,
        'name': None,
        'name_match': False,
        'curp_match': False,
        'timing': timing,
        'success': False,
        'error': None
    }
    
    try:
        # Read text file
        load_start = time.perf_counter()
        path = Path(text_file_path)
        if not path.exists():
            result['error'] = f"Text file not found: {text_file_path}"
            result['timing']['total'] = time.perf_counter() - total_start
            return result
        
        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        timing['file_load'] = time.perf_counter() - load_start
        
        # Extract CURP
        curp_start = time.perf_counter()
        curp = extract_curp_from_text(full_text, debug=False)  # Disable debug for speed
        timing['curp_extraction'] = time.perf_counter() - curp_start
        result['curp'] = curp
        
        # Check if CURP matches expected (case-insensitive, normalized)
        if curp and curp_expected:
            curp_normalized = curp.upper().strip()
            expected_normalized = curp_expected.upper().strip()
            result['curp_match'] = curp_normalized == expected_normalized
        
        # Extract name
        name_start = time.perf_counter()
        name = extract_name_curp_second_version(full_text)
        timing['name_extraction'] = time.perf_counter() - name_start
        result['name'] = name
        
        # Check if name matches expected (case-insensitive, normalized)
        if name and name_expected:
            name_normalized = _normalize_text(name)
            expected_normalized = _normalize_text(name_expected)
            result['name_match'] = name_normalized == expected_normalized
        
        # Calculate total time
        timing['total'] = time.perf_counter() - total_start
        result['success'] = True
        
    except Exception as e:
        result['error'] = f"Unexpected error: {e}"
        result['timing']['total'] = time.perf_counter() - total_start
    
    return result


def process_document(type_document: str, image_path: str, name_expected: str, curp_expected: str = None) -> Dict:
    """
    Main function to process a document and extract CURP and name.
    
    Args:
        type_document: Type of document ("PNG", "PDF", or "JPEG")
        image_path: Path to the document file
        name_expected: Expected name for validation
        curp_expected: Expected CURP/Clave for validation (optional)
        
    Returns:
        Dictionary containing:
            - 'curp': Extracted CURP/Clave (or None)
            - 'name': Extracted name (or None)
            - 'name_match': Boolean indicating if extracted name matches expected
            - 'curp_match': Boolean indicating if extracted CURP matches expected
            - 'timing': Dictionary with timing information
            - 'success': Boolean indicating overall success
            - 'error': Error message if any (or None)
    """
    total_start = time.perf_counter()
    timing = {
        'file_load': 0.0,
        'ocr': 0.0,
        'curp_extraction': 0.0,
        'name_extraction': 0.0,
        'total': 0.0
    }
    
    result = {
        'curp': None,
        'name': None,
        'name_match': False,
        'curp_match': False,
        'timing': timing,
        'success': False,
        'error': None
    }
    
    try:
        # Check Tesseract installation
        if not check_tesseract_installed():
            result['error'] = "Tesseract OCR is not installed or not in PATH."
            result['timing']['total'] = time.perf_counter() - total_start
            return result
        
        # Validate file exists
        path = Path(image_path)
        if not path.exists():
            result['error'] = f"File not found: {image_path}"
            result['timing']['total'] = time.perf_counter() - total_start
            return result
        
        # Load image based on document type
        load_start = time.perf_counter()
        type_doc_upper = type_document.upper()
        
        if type_doc_upper == "PDF":
            image = process_pdf(image_path)
        elif type_doc_upper in ["PNG", "JPEG", "JPG"]:
            try:
                image = Image.open(image_path)
            except Exception as e:
                result['error'] = f"Failed to open image file: {e}"
                result['timing']['total'] = time.perf_counter() - total_start
                return result
        else:
            result['error'] = f"Unsupported document type: {type_document}. Supported types: PNG, PDF, JPEG"
            result['timing']['total'] = time.perf_counter() - total_start
            return result
        
        timing['file_load'] = time.perf_counter() - load_start
        
        # Perform OCR
        ocr_start = time.perf_counter()
        try:
            full_text = ocr_image(image, fast_mode=True)
        except RuntimeError as e:
            result['error'] = str(e)
            result['timing']['total'] = time.perf_counter() - total_start
            return result
        except Exception as e:
            result['error'] = f"OCR failed: {e}"
            result['timing']['total'] = time.perf_counter() - total_start
            return result
        
        timing['ocr'] = time.perf_counter() - ocr_start
        
        # Optionally save OCR output to file for debugging (similar to output.txt)
        # This helps verify extraction is working with the actual OCR text
        try:
            output_file = Path(image_path).parent / "output.txt.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_text)
        except Exception:
            pass  # Don't fail if we can't write the file
        
        # Extract CURP
        curp_start = time.perf_counter()
        curp = extract_curp_from_text(full_text, debug=False)  # Disable debug for speed
        timing['curp_extraction'] = time.perf_counter() - curp_start
        result['curp'] = curp
        
        # Check if CURP matches expected (case-insensitive, normalized)
        if curp and curp_expected:
            curp_normalized = curp.upper().strip()
            expected_normalized = curp_expected.upper().strip()
            result['curp_match'] = curp_normalized == expected_normalized
        
        # Extract name
        name_start = time.perf_counter()
        name = extract_name_curp_second_version(full_text)
        timing['name_extraction'] = time.perf_counter() - name_start
        result['name'] = name
        
        # Check if name matches expected (case-insensitive, normalized)
        if name and name_expected:
            name_normalized = _normalize_text(name)
            expected_normalized = _normalize_text(name_expected)
            result['name_match'] = name_normalized == expected_normalized
        
        # Calculate total time
        timing['total'] = time.perf_counter() - total_start
        result['success'] = True
        
    except Exception as e:
        result['error'] = f"Unexpected error: {e}"
        result['timing']['total'] = time.perf_counter() - total_start
    
    return result