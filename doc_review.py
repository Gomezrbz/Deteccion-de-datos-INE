import io
import re
import sys
import unicodedata
import argparse
from pathlib import Path
from typing import Optional, TypedDict, Literal
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

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
    Preprocess OpenCV image array for OCR.
    mode: 'binarize' (Otsu), 'adapt' (adaptive threshold)
    """
    # Mild denoise
    gray = cv2.GaussianBlur(img_array, (3, 3), 0)
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

    # Perform OCR
    try:
        # Two preprocessing passes; whichever yields better-looking text wins.
        bin_otsu = _preprocess_for_ocr(gray, mode="binarize")
        bin_adapt = _preprocess_for_ocr(gray, mode="adapt")
        t1 = pytesseract.image_to_string(bin_otsu, lang=OCR_LANG, config=OCR_CONFIG)
        t2 = pytesseract.image_to_string(bin_adapt, lang=OCR_LANG, config=OCR_CONFIG)
        text = t1 if _score_ocr_text(t1) >= _score_ocr_text(t2) else t2
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
    return text


def ocr_image(image: Image.Image) -> str:
    """
    OCR with rotation attempts (useful for sideways photos).
    """
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


def detect_document_type(text: str) -> DocType:
    t = _normalize_text(text)
    if "INSTITUTO NACIONAL ELECTORAL" in t or "CREDENCIAL PARA VOTAR" in t or "CLAVE DE ELECTOR" in t:
        return "INE"
    # Passport (MRZ is a strong signal)
    if "PASAPORTE" in t or re.search(r"\bP<", t):
        return "PASSPORT"
    if "CONSTANCIA" in t and ("REGISTRO DE POBLACION" in t or "CURP" in t or "RENAPO" in t):
        return "CURP"
    return "UNKNOWN"


def _clean_name_piece(s: str) -> str:
    s = _strip_accents(s).upper()
    s = re.sub(r"[^A-Z ]+", " ", s)
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
    )
    if any(b in l for b in banned):
        return False
    # Names are typically mostly letters/spaces
    return bool(re.search(r"[A-Z]{2,}", l))


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
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    upper_lines = [_normalize_text(l) for l in lines]

    for i, line in enumerate(upper_lines):
        if "NOMBRE" in line:
            # Case A: name appears on same line after keyword
            same = re.split(r"NOMBRE[: ]*", lines[i], flags=re.IGNORECASE)
            if len(same) == 2 and _looks_like_name_line(same[1]):
                return _clean_name_piece(same[1])

            # Case B: common INE layout:
            # NOMBRE
            # APELLIDO_PATERNO
            # APELLIDO_MATERNO
            # NOMBRES
            candidates: list[str] = []
            for j in range(i + 1, min(i + 6, len(lines))):
                if _looks_like_name_line(lines[j]):
                    candidates.append(_clean_name_piece(lines[j]))
                if len(candidates) >= 3:
                    break
            if len(candidates) >= 3:
                ap_pat, ap_mat, nombres = candidates[0], candidates[1], candidates[2]
                return _clean_name_piece(f"{nombres} {ap_pat} {ap_mat}")
            if candidates:
                return candidates[0]
    return None


def extract_name_curp(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    upper_lines = [_normalize_text(l) for l in lines]
    for i, line in enumerate(upper_lines):
        if re.search(r"\bNOMBRE\b", line):
            # Often: "Nombre" then next line is "RAUL GOMEZ BERMUDEZ"
            if i + 1 < len(lines) and _looks_like_name_line(lines[i + 1]):
                return _clean_name_piece(lines[i + 1])
            # Sometimes: "Nombre: ..."
            m = re.search(r"\bNOMBRE\b[: ]+(.*)$", lines[i], flags=re.IGNORECASE)
            if m and _looks_like_name_line(m.group(1)):
                return _clean_name_piece(m.group(1))
    # Fallback: try to find any long all-caps line that looks like a full name
    for line in lines:
        l = _clean_name_piece(line)
        if len(l.split()) >= 3 and _looks_like_name_line(l):
            return l
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
            doc_type_first = detect_document_type(full_text)
            name_first = extract_name_best(full_text, doc_type_first)
            if name_first is None and len(images) > pages_to_process:
                remaining_text = "\n".join(
                    ocr_image(img) for img in images[pages_to_process:]
                )
                full_text += "\n" + remaining_text
            
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


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Extract the person's name from INE / Passport / CURP and suggest a filename."
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=f"Path to an image (jpg/png) or PDF. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES_TO_PROCESS,
        help="Max PDF pages to OCR (default: 3).",
    )
    parser.add_argument(
        "--print-text",
        action="store_true",
        help="Print raw OCR text (debug).",
    )
    args = parser.parse_args()

    # Check if Tesseract is installed before processing
    if not check_tesseract_installed():
        print(
            "ERROR: Tesseract OCR is not installed or not in your PATH.\n"
            "Install on Ubuntu/Debian:\n"
            "  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-spa\n\n"
            "Install on macOS:\n"
            "  brew install tesseract tesseract-lang\n",
            file=sys.stderr,
        )
        sys.exit(1)

    file_path = Path(args.path)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(2)

    try:
        result = get_name_from_path(file_path, max_pages=args.max_pages)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(3)

    # Primary output: suggested filename (or fallback)
    suggested = result["suggested_filename"] or file_path.name
    print(suggested)

    if args.print_text:
        print("\n=== DOCUMENT TYPE ===")
        print(result["document_type"])
        print("\n=== EXTRACTED NAME ===")
        print(result["extracted_name"] or "Name not found")
        print("\n=== RAW TEXT (first 1200 chars) ===")
        print(result["raw_text"][:1200])


if __name__ == "__main__":
    main()
