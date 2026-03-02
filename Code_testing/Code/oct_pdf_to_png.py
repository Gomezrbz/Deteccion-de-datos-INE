from pdf2image import convert_from_path
import os
from pathlib import Path
from typing import Optional, List


def pdf_to_png(pdf_path: str, output_dir: str, poppler_path: Optional[str] = None) -> List[str]:
    """
    Convert PDF to PNG images.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save PNG files
        poppler_path: Optional path to poppler bin directory. If None, tries to find it relative to this script.
    
    Returns:
        List of paths to generated PNG files (first page only for batch processing)
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        RuntimeError: If poppler is not found or conversion fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Extract filename without extension
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine poppler path
    if poppler_path is None:
        # Try relative to this script's location
        script_dir = Path(__file__).parent
        default_poppler = script_dir / "poppler-25.12.0" / "Library" / "bin"
        if default_poppler.exists():
            poppler_path = str(default_poppler)
        else:
            poppler_path = None  # Let pdf2image try system PATH
    
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=400,
            fmt="png",
            grayscale=True,
            poppler_path=poppler_path if poppler_path else None
        )
    except Exception as e:
        error_msg = str(e)
        if "poppler" in error_msg.lower() or "Unable to get page count" in error_msg:
            raise RuntimeError(
                f"Poppler not found. Please ensure poppler is installed or set poppler_path. Error: {error_msg}"
            ) from e
        raise RuntimeError(f"Failed to convert PDF to PNG: {error_msg}") from e
    
    output_paths = []
    
    # For batch processing, only return first page
    if pages:
        output_path = os.path.join(output_dir, f"{pdf_filename}_page_1.png")
        pages[0].save(output_path, "PNG")
        output_paths.append(output_path)
        
        # Log warning if multiple pages
        if len(pages) > 1:
            print(f"Warning: PDF has {len(pages)} pages, processing only first page: {pdf_path}")
    
    return output_paths


if __name__ == "__main__":
    # Original hardcoded logic for standalone execution
    pdf_path = "../Test_Data/pdf/RAFAEL LOPEZ LINARES.pdf"
    output_dir = "../Test_Data/png"
    
    output_paths = pdf_to_png(pdf_path, output_dir)
    print(f"Converted {len(output_paths)} page(s) from {pdf_path}")