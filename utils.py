import re
import sys
import time
import unicodedata
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Dict, List
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path


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


_CURP_RE = re.compile(r"^[A-Z]{4}\d{6}[A-Z]{6}\d{2}$", re.IGNORECASE)


def dob_from_curp(curp: str) -> date:
    """
    Extract DOB from CURP and return a datetime.date.

    CURP format includes DOB as YYMMDD starting at position 4 (0-based).
    Century is inferred using a pivot rule based on the current year:
    - if YY > current_YY -> 1900s
    - else -> 2000s
    
    Args:
        curp: CURP string (18 characters)
        
    Returns:
        datetime.date object with the date of birth
        
    Raises:
        ValueError: If CURP format is invalid
    """
    curp = curp.strip().upper()

    # Basic validation (keeps it practical; CURP has more rules, but this is enough for DOB extraction)
    if not _CURP_RE.match(curp):
        raise ValueError(f"Invalid CURP format: {curp}")

    yymmdd = curp[4:10]  # YYMMDD
    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])

    current_yy = int(datetime.now().strftime("%y"))
    year = (1900 + yy) if yy > current_yy else (2000 + yy)

    # Validates real dates too (e.g., rejects 990231)
    return date(year, mm, dd)


def dob_from_curp_str(curp: str) -> Optional[str]:
    """
    Returns DOB as 'DD/MM/YYYY' string, or None if CURP is invalid.
    
    Args:
        curp: CURP string
        
    Returns:
        Date of birth as 'DD/MM/YYYY' string, or None if extraction fails
    """
    try:
        d = dob_from_curp(curp)
        return d.strftime("%d/%m/%Y")
    except (ValueError, IndexError, TypeError):
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


def pdf_to_png(pdf_path: str, dpi: int = 400) -> List[str]:
    """
    Converts a PDF into PNG images with enhanced preprocessing for better OCR.
    One PNG per page.
    Saves images in the same directory as the PDF.

    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for conversion (default 400, increased for better OCR quality)
        
    Returns:
        List of generated PNG paths
        
    Raises:
        ValueError: If file is not a PDF
        FileNotFoundError: If PDF file doesn't exist
        RuntimeError: If poppler is not installed
    """
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    directory, filename = os.path.split(pdf_path)
    name, _ = os.path.splitext(filename)

    try:
        # Convert PDF to images with higher DPI for better quality
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        error_msg = str(e)
        if "poppler" in error_msg.lower() or "Unable to get page count" in error_msg:
            poppler_instructions = (
                "\n\n" + "=" * 80 + "\n"
                "ERROR: Poppler is not installed or not in PATH.\n"
                "pdf2image requires Poppler to convert PDF files.\n\n"
                "To install Poppler:\n"
                "  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/\n"
                "           Extract and add the 'bin' folder to your PATH environment variable\n\n"
                "  macOS:   brew install poppler\n\n"
                "  Linux:   sudo apt-get install poppler-utils  (Ubuntu/Debian)\n"
                "           sudo yum install poppler-utils       (CentOS/RHEL)\n"
                + "=" * 80
            )
            raise RuntimeError(poppler_instructions) from e
        else:
            raise RuntimeError(f"Failed to convert PDF to PNG: {error_msg}") from e

    output_paths = []

    for i, page in enumerate(pages, start=1):
        # Ensure image is in RGB mode (required for some operations)
        if page.mode != 'RGB':
            page = page.convert('RGB')
        
        # Enhance image for better OCR results
        try:
            # 1. Enhance contrast (helps with text clarity)
            enhancer = ImageEnhance.Contrast(page)
            page = enhancer.enhance(1.2)  # Increase contrast by 20%
            
            # 2. Enhance sharpness (helps with text edge definition)
            enhancer = ImageEnhance.Sharpness(page)
            page = enhancer.enhance(1.3)  # Increase sharpness by 30%
            
            # 3. Apply slight unsharp mask filter for better text definition
            page = page.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        except Exception as e:
            # If enhancement fails, continue with original image
            print(f"Warning: Image enhancement failed for page {i}: {e}")
        
        # Generate output path - use same name as PDF but with .png extension for first page
        if i == 1:
            output_path = os.path.join(directory, f"{name}.png")
        else:
            output_path = os.path.join(directory, f"{name}_page_{i}.png")
        
        # Save with high quality settings
        page.save(output_path, "PNG", optimize=False, compress_level=1)
        output_paths.append(output_path)

    return output_paths


def process_text_file(text_file_path: str, name_expected: str, curp_expected: str = None, dob_expected: str = None) -> Dict:
    """
    Process a text file directly (faster than OCR).
    This is useful when you already have the OCR output from tesseract command.
    
    Args:
        text_file_path: Path to the text file (e.g., output.txt.txt from tesseract)
        name_expected: Expected name for validation
        curp_expected: Expected CURP/Clave for validation (optional)
        dob_expected: Expected date of birth in 'DD/MM/YYYY' format (optional)
        
    Returns:
        Dictionary containing:
            - 'curp': Extracted CURP/Clave (or None)
            - 'name': Extracted name (or None)
            - 'dob': Extracted date of birth in 'DD/MM/YYYY' format (or None)
            - 'name_match': Boolean indicating if extracted name matches expected
            - 'curp_match': Boolean indicating if extracted CURP matches expected
            - 'dob_match': Boolean indicating if extracted DOB matches expected
            - 'timing': Dictionary with timing information
            - 'success': Boolean indicating overall success
            - 'error': Error message if any (or None)
    """
    total_start = time.perf_counter()
    timing = {
        'file_load': 0.0,
        'curp_extraction': 0.0,
        'dob_extraction': 0.0,
        'name_extraction': 0.0,
        'total': 0.0
    }
    
    result = {
        'curp': None,
        'name': None,
        'dob': None,
        'name_match': False,
        'curp_match': False,
        'dob_match': False,
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
        
        # Extract DOB from CURP
        dob_start = time.perf_counter()
        dob = None
        if curp:
            dob = dob_from_curp_str(curp)
        timing['dob_extraction'] = time.perf_counter() - dob_start
        result['dob'] = dob
        
        # Check if DOB matches expected
        if dob and dob_expected:
            result['dob_match'] = dob == dob_expected
        
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


def manage_output_file(output_file_path: str, curp_match: bool, name_match: bool) -> None:
    """
    Manage output file based on extraction results.
    
    If both CURP and Name match: delete the output file.
    If either doesn't match: rename the output file to error_output_file.
    
    Args:
        output_file_path: Path to the output file (e.g., output.txt)
        curp_match: Boolean indicating if CURP matches expected
        name_match: Boolean indicating if name matches expected
    """
    output_path = Path(output_file_path)
    
    if not output_path.exists():
        print(f"⚠ Output file not found: {output_path}")
        return
    
    both_match = curp_match and name_match
    
    print("=" * 80)
    print("Output File Management")
    print("=" * 80)
    
    if both_match:
        # Both match - delete the output file
        try:
            output_path.unlink()
            print(f"✓ Output file deleted (both CURP and Name matched): {output_path}")
        except Exception as e:
            print(f"✗ Error deleting output file: {e}")
    else:
        # One or both don't match - rename to error_output_file
        try:
            # Preserve file extension
            file_extension = output_path.suffix
            error_file = output_path.parent / f"error_output_file{file_extension}"
            
            # If error file already exists, add a number
            counter = 1
            while error_file.exists():
                error_file = output_path.parent / f"error_output_file_{counter}{file_extension}"
                counter += 1
            
            output_path.rename(error_file)
            print(f"✓ Output file renamed (CURP or Name mismatch): {output_path.name} -> {error_file.name}")
        except Exception as e:
            print(f"✗ Error renaming output file: {e}")
    
    print("=" * 80)
    print()