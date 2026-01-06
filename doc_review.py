import sys
import argparse
from pathlib import Path
from methods import (
    check_tesseract_installed,
    get_name_from_path,
    MAX_PAGES_TO_PROCESS,
)


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Extract the person's name from INE / Passport / CURP and suggest a filename."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to an image (jpg/png) or PDF file to review.",
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

    # Validate file path
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(2)
    
    if not file_path.is_file():
        print(f"Error: '{file_path}' is not a file.", file=sys.stderr)
        sys.exit(2)

    # Check if Tesseract is installed before processing
    if not check_tesseract_installed():
        print(
            "ERROR: Tesseract OCR is not installed or not in your PATH.\n"
            "Install on Ubuntu/Debian:\n"
            "  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-spa\n\n"
            "Install on macOS:\n"
            "  brew install tesseract tesseract-lang\n\n"
            "Install on Windows:\n"
            "  Download from: https://github.com/UB-Mannheim/tesseract/wiki",
            file=sys.stderr,
        )
        sys.exit(1)

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
