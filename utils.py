import io
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional, TypedDict, Literal
import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# Constants
SUPPORTED_IMAGE_EXTS = {"jpg", "jpeg", "png"}
SUPPORTED_PDF_EXT = "pdf"
PDF_DPI = 300
OCR_LANG = "spa+eng"
OCR_CONFIG = "--psm 6 -c preserve_interword_spaces=1"
MAX_PAGES_TO_PROCESS = 3  # Process first 3 pages for efficiency
DocType = Literal["INE", "PASSPORT", "CURP", "UNKNOWN"]

# If you don't want to pass the path in the command line,
# set it here and just run: `python3 doc_review.py`
DEFAULT_INPUT_PATH = "files/Vertical ID.jpeg"


class ExtractionResult(TypedDict):
    """Type definition for extraction result."""
    extracted_name: Optional[str]
    document_type: DocType
    suggested_filename: Optional[str]
    raw_text: str


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


def _score_ocr_text(text: str) -> int:
    t = _normalize_text(text)
    score = 0
    # Prefer texts that look like the target docs
    for kw in (
        "INSTITUTO NACIONAL ELECTORAL",
        "CREDENCIAL PARA VOTAR",
        "CLAVE DE ELECTOR",
        "PASAPORTE",
        "P<",
        "CURP",
        "REGISTRO DE POBLACION",
        "CONSTANCIA",
        "NOMBRE",
    ):
        if kw in t:
            score += 50
    # Favor "denser" text (more letters)
    score += len(re.findall(r"[A-Z]", t))
    return score


def _ocr_image_single(image: Image.Image) -> str:
    """
    Perform OCR on a PIL Image with enhanced preprocessing.
    Tries multiple preprocessing methods and OCR configs to get the best result.
    
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

    # Perform OCR with multiple preprocessing methods and configs
    candidates = []
    
    try:
        # Try different preprocessing methods
        preprocess_methods = ["enhanced", "binarize", "adapt"]
        ocr_configs = [
            OCR_CONFIG,  # Original config
            "--psm 6 -c preserve_interword_spaces=1",  # Default
            "--psm 11 -c preserve_interword_spaces=1",  # Sparse text
            "--psm 12 -c preserve_interword_spaces=1",  # Single uniform block
            "--psm 4 -c preserve_interword_spaces=1",   # Single column
        ]
        
        for method in preprocess_methods:
            processed = _preprocess_for_ocr(gray, mode=method)
            processed_img = Image.fromarray(processed)
            
            # Try different OCR configs
            for config in ocr_configs[:3]:  # Limit to first 3 configs to avoid too many attempts
                try:
                    text = pytesseract.image_to_string(processed_img, lang=OCR_LANG, config=config)
                    if text.strip():
                        candidates.append((text, _score_ocr_text(text)))
                except Exception:
                    continue
        
        # If no candidates, try original method as fallback
        if not candidates:
            bin_otsu = _preprocess_for_ocr(gray, mode="binarize")
            bin_adapt = _preprocess_for_ocr(gray, mode="adapt")
            t1 = pytesseract.image_to_string(Image.fromarray(bin_otsu), lang=OCR_LANG, config=OCR_CONFIG)
            t2 = pytesseract.image_to_string(Image.fromarray(bin_adapt), lang=OCR_LANG, config=OCR_CONFIG)
            candidates = [(t1, _score_ocr_text(t1)), (t2, _score_ocr_text(t2))]
        
        # Return the best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        else:
            return ""
            
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
    
    # Standard mode: try multiple rotations
    candidates: list[str] = []
    for angle in (0, 90, 180, 270):
        img = image.rotate(angle, expand=True) if angle else image
        try:
            candidates.append(_ocr_image_single(img))
        except RuntimeError:
            # Bubble up "tesseract not installed" errors
            raise
        except Exception:
            continue
    if not candidates:
        return ""
    return max(candidates, key=_score_ocr_text)


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


def detect_document_type(text: str) -> DocType:
    t = _normalize_text(text)
    
    # INE detection - try multiple variations to handle OCR errors
    ine_keywords = [
        "INSTITUTO NACIONAL ELECTORAL",
        "INSTITUTO NACIONAL ELECTOFAL",  # OCR error
        "CREDENCIAL PARA VOTAR",
        "CREDENCIAL PAFA VOTAR",  # OCR error
        "CLAVE DE ELECTOR",
        "CLAVE DE ELECTOF",  # OCR error
    ]
    if any(kw in t for kw in ine_keywords):
        return "INE"
    
    # Passport detection - MRZ is a strong signal
    if "PASAPORTE" in t or "PASAPOFTE" in t or re.search(r"\bP<", t):
        return "PASSPORT"
    
    # CURP detection
    curp_keywords = ["CURP", "CONSTANCIA"]
    curp_context = ["REGISTRO DE POBLACION", "RENAPO", "REGISTRO"]
    if any(kw in t for kw in curp_keywords) and any(ctx in t for ctx in curp_context):
        return "CURP"
    
    return "UNKNOWN"


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


def extract_name_from_mrz(text: str) -> Optional[str]:
    """
    Extract name from passport MRZ, e.g.:
      P<MEXGOMEZ<BERMUDEZ<<RAUL<<<<<<<<<<<<<<
    """
    t = _normalize_text(text)
    m = re.search(r"\bP<([A-Z]{3})([A-Z<]{5,})\b", t)
    if not m:
        return None
    rest = m.group(2)
    # Split surname(s) and given names
    parts = rest.split("<<", 1)
    if len(parts) != 2:
        return None
    surnames_raw, given_raw = parts[0], parts[1]
    surnames = " ".join([p for p in surnames_raw.split("<") if p])
    given = " ".join([p for p in given_raw.split("<") if p])
    full = _clean_name_piece(f"{given} {surnames}")
    return full or None


def extract_name_ine(text: str) -> Optional[str]:
    """
    Extract name from INE document.
    INE format: NOMBRE is followed by:
    - Line 1: Father's last name (Apellido Paterno)
    - Line 2: Mother's last name (Apellido Materno)
    - Line 3: First name(s) (Nombre(s))
    
    We want output: First Name(s) + Father Last Name + Mother Last Name
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    upper_lines = [_normalize_text(l) for l in lines]

    # Find NOMBRE and DOMICILIO positions
    nombre_idx = None
    domicilio_idx = None
    
    for i, line in enumerate(upper_lines):
        # Look for "NOMBRE" keyword (handle OCR errors)
        if ("NOMBRE" in line or "NOMBPE" in line) and nombre_idx is None:
            nombre_idx = i
        # Look for "DOMICILIO" to know where name section ends
        if "DOMICILIO" in line and domicilio_idx is None:
            domicilio_idx = i
            break  # Stop at first DOMICILIO
    
    if nombre_idx is None:
        # Fallback: try to find name lines without NOMBRE keyword
        for line in lines:
            cleaned = _clean_name_piece(line)
            words = cleaned.split()
            if len(words) >= 3 and _looks_like_name_line(line):
                return cleaned
        return None
    
    # Extract lines between NOMBRE and DOMICILIO (or next 5 lines if DOMICILIO not found)
    end_idx = domicilio_idx if domicilio_idx else min(nombre_idx + 6, len(lines))
    name_candidates = []
    
    # Collect potential name lines between NOMBRE and DOMICILIO
    for j in range(nombre_idx + 1, end_idx):
        if j >= len(lines):
            break
        
        line = lines[j]
        cleaned = _clean_name_piece(line)
        
        # Skip if it's a label or empty
        if not cleaned or len(cleaned) < 2:
            continue
        
        # Skip if it looks like a label (DOMICILIO, etc.)
        if any(label in cleaned for label in ["DOMICILIO", "CLAVE", "CURP", "FECHA", "SEXO"]):
            break
        
        # Check if it looks like a name component
        words = cleaned.split()
        if words and all(len(w) >= 2 and re.match(r'^[A-Z]+$', w) for w in words):
            name_candidates.append(cleaned)
    
    # INE format: typically 3 lines (Father Last, Mother Last, First Name)
    # But sometimes it can be 2 lines or combined
    if len(name_candidates) >= 3:
        # Standard format: [Father Last, Mother Last, First Name]
        ap_paterno = name_candidates[0]
        ap_materno = name_candidates[1]
        nombres = name_candidates[2]
        # Reorder: First Name + Father Last + Mother Last
        return f"{nombres} {ap_paterno} {ap_materno}"
    elif len(name_candidates) == 2:
        # Sometimes first name is on same line or missing one last name
        # Try to identify which is which based on position
        # Usually: [Last Name(s), First Name] or [Father Last, Mother Last]
        # If both look like last names (shorter), combine them
        if all(len(c.split()) <= 2 for c in name_candidates):
            # Both are likely last names, check if there's a first name elsewhere
            return " ".join(name_candidates)
        else:
            # One is likely a full name, return as is
            return " ".join(name_candidates)
    elif len(name_candidates) == 1:
        # Single line - might be full name or just one component
        return name_candidates[0]
    
    # Fallback: try to find name on same line as NOMBRE
    nombre_line = lines[nombre_idx]
    patterns = [r"NOMBRE[: ]*", r"NOMBPE[: ]*"]
    for pattern in patterns:
        parts = re.split(pattern, nombre_line, flags=re.IGNORECASE)
        if len(parts) == 2 and _looks_like_name_line(parts[1]):
            return _clean_name_piece(parts[1])
    
    return None


def extract_name_curp(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    upper_lines = [_normalize_text(l) for l in lines]
    
    for i, line in enumerate(upper_lines):
        # Look for "NOMBRE" with OCR error tolerance
        if re.search(r"\bNOMBPE\b", line) or re.search(r"\bNOMBRE\b", line):
            # Often: "Nombre" then next line is "RAUL GOMEZ BERMUDEZ"
            if i + 1 < len(lines) and _looks_like_name_line(lines[i + 1]):
                return _clean_name_piece(lines[i + 1])
            # Sometimes: "Nombre: ..."
            patterns = [r"\bNOMBPE\b[: ]+(.*)$", r"\bNOMBRE\b[: ]+(.*)$"]
            for pattern in patterns:
                m = re.search(pattern, lines[i], flags=re.IGNORECASE)
                if m and _looks_like_name_line(m.group(1)):
                    return _clean_name_piece(m.group(1))
    
    # Fallback: try to find any long all-caps line that looks like a full name
    for line in lines:
        cleaned = _clean_name_piece(line)
        words = cleaned.split()
        if len(words) >= 3 and _looks_like_name_line(line):
            return cleaned
    
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


def extract_name_passport(text: str) -> Optional[str]:
    # Best: parse MRZ
    mrz = extract_name_from_mrz(text)
    if mrz:
        return mrz

    # Otherwise: "Apellidos" + "Nombres" blocks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    upper_lines = [_normalize_text(l) for l in lines]

    surname: Optional[str] = None
    given: Optional[str] = None
    for i, line in enumerate(upper_lines):
        if surname is None and ("APELLIDOS" in line or "SURNAME" in line):
            if i + 1 < len(lines) and _looks_like_name_line(lines[i + 1]):
                surname = _clean_name_piece(lines[i + 1])
        if given is None and (re.search(r"\bNOMBRES\b", line) or "GIVEN NAMES" in line or re.search(r"\bNOMBRE\b", line)):
            if i + 1 < len(lines) and _looks_like_name_line(lines[i + 1]):
                given = _clean_name_piece(lines[i + 1])
        if surname and given:
            break
    if surname and given:
        return _clean_name_piece(f"{given} {surname}")
    return None


def extract_name_best(text: str, doc_type: DocType) -> Optional[str]:
    if doc_type == "PASSPORT":
        return extract_name_passport(text)
    if doc_type == "CURP":
        return extract_name_curp(text)
    if doc_type == "INE":
        return extract_name_ine(text)
    # Unknown: try all strategies
    for fn in (extract_name_passport, extract_name_curp, extract_name_ine):
        name = fn(text)
        if name:
            return name
    return None


def suggest_filename(doc_type: DocType, name: Optional[str], original_suffix: str) -> Optional[str]:
    if not name:
        return None
    safe = _clean_name_piece(name).replace(" ", "_")
    prefix = doc_type if doc_type != "UNKNOWN" else "DOC"
    return f"{prefix}_{safe}{original_suffix}"


def extract_name_ine_from_image(image: Image.Image) -> Optional[str]:
    """
    Extract name from INE document image (PNG/JPG).
    Specifically designed for INE format where name appears between NOMBRE and DOMICILIO.
    """
    full_text = ocr_image(image)
    return extract_name_ine(full_text)


def extract_name_ine_from_pdf(file_bytes: bytes, max_pages: int = MAX_PAGES_TO_PROCESS) -> Optional[str]:
    """
    Extract name from INE PDF document.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    zoom = PDF_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    # Process first page (INE is usually single page or name is on first page)
    page = doc[0]
    pix = page.get_pixmap(matrix=matrix)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    full_text = ocr_image(img)
    
    doc.close()
    return extract_name_ine(full_text)


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
            # Open PDF from bytes using PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            
            # Calculate zoom factor for desired DPI (72 is default DPI)
            zoom = PDF_DPI / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            
            # Process only first N pages for efficiency
            pages_to_process = min(total_pages, max_pages)
            images = []
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix)
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            
            # Use list comprehension and join for better performance
            text_parts = [ocr_image(img) for img in images]
            full_text = "\n".join(text_parts)
            
            # If name not found in first pages and we have more pages, process remaining
            doc_type_first = detect_document_type(full_text)
            name_first = extract_name_best(full_text, doc_type_first)
            if name_first is None and total_pages > pages_to_process:
                remaining_images = []
                for page_num in range(pages_to_process, total_pages):
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=matrix)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    remaining_images.append(img)
                remaining_text = "\n".join(
                    ocr_image(img) for img in remaining_images
                )
                full_text += "\n" + remaining_text
            
            doc.close()
            
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {e}") from e
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_IMAGE_EXTS)}, {SUPPORTED_PDF_EXT}")

    # Extract doc type + name from full text
    doc_type = detect_document_type(full_text)
    name = extract_name_best(full_text, doc_type)
    suggested = suggest_filename(doc_type, name, Path(filename).suffix.lower())

    return {
        "extracted_name": name,
        "document_type": doc_type,
        "suggested_filename": suggested,
        "raw_text": full_text
    }


def get_name_from_path(path: Path, max_pages: int = MAX_PAGES_TO_PROCESS) -> ExtractionResult:
    return get_name_from_file(path.read_bytes(), path.name, max_pages=max_pages)
