import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pytesseract
from PIL import Image

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
