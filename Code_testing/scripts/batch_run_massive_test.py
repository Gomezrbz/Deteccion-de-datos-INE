"""
Batch OCR Test Pipeline

This script processes all PDFs from the Massive Test folder through the OCR pipeline:
1. Convert PDF to PNG (first page only)
2. Optimize image for OCR (extract card, nombre ROI, nombre OCR)
3. Process rectangles on nombre ROI and extract name (between NOMBRE and DOMICILIO)
4. Validate against ground truth in data.txt
5. Generate CSV report and terminal output showing mismatches

Usage:
    python batch_run_massive_test.py
    python batch_run_massive_test.py --dry-run

Output:
    - Test_Results/<pdf_basename>/ - per-PDF folder with all artifacts
    - Test_Results/summary_nombre_validation.csv - summary report
    - Terminal output showing names NOT found in data.txt
"""

import sys
import os
import argparse
import csv
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add Code directory to path to import refactored functions
SCRIPT_DIR = Path(__file__).parent
CODE_DIR = SCRIPT_DIR.parent / "Code"
sys.path.insert(0, str(CODE_DIR))

try:
    from oct_pdf_to_png import pdf_to_png
    from oct_opt_image import process_image_for_ocr
    from oct_rectangles import process_rectangles
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please ensure all scripts are in Code_testing/Code/")
    sys.exit(1)

# Absolute paths as constants
PDF_INPUT_DIR = Path(r"C:\Projects\Deteccion-de-datos-INE\Code_testing\Test_Data\pdf\Massive Test")
GROUND_TRUTH_FILE = PDF_INPUT_DIR / "data.txt"
TEST_RESULTS_DIR = Path(r"C:\Projects\Deteccion-de-datos-INE\Code_testing\Test_Data\Test_Results")
SUMMARY_CSV = TEST_RESULTS_DIR / "summary_nombre_validation.csv"


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for matching: uppercase, strip accents, collapse whitespace, remove punctuation.
    
    Args:
        text: Input text string
    
    Returns:
        Normalized string
    """
    if not text:
        return ""
    
    # Strip accents/diacritics
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    
    # Uppercase
    text = text.upper()
    
    # Remove punctuation, keep only letters and spaces
    text = re.sub(r"[^A-Z ]+", " ", text)
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def match_name(extracted: Optional[str], ground_truth_lines: List[str]) -> Tuple[bool, Optional[str], bool]:
    """
    Match extracted name against ground truth lines.
    
    Args:
        extracted: Extracted name string (may be None)
        ground_truth_lines: List of ground truth name strings
    
    Returns:
        Tuple of (found, matched_line, strict_match)
        - found: True if match found (substring or strict)
        - matched_line: The matched ground truth line (or None)
        - strict_match: True if exact match after normalization
    """
    if not extracted:
        return False, None, False
    
    extracted_norm = normalize_for_matching(extracted)
    
    if not extracted_norm:
        return False, None, False
    
    for gt_line in ground_truth_lines:
        gt_norm = normalize_for_matching(gt_line)
        
        if not gt_norm:
            continue
        
        # Strict match
        if extracted_norm == gt_norm:
            return True, gt_line, True
        
        # Substring matching (bidirectional)
        if extracted_norm in gt_norm or gt_norm in extracted_norm:
            return True, gt_line, False
    
    return False, None, False


def ensure_unique_folder(base_path: Path) -> Path:
    """
    Ensure folder path is unique by appending counter if needed.
    
    Args:
        base_path: Base folder path
    
    Returns:
        Unique folder path
    """
    if not base_path.exists():
        return base_path
    
    counter = 1
    while True:
        new_path = base_path.parent / f"{base_path.name}_{counter}"
        if not new_path.exists():
            return new_path
        counter += 1


def load_ground_truth(ground_truth_path: Path) -> List[str]:
    """
    Load ground truth names from data.txt file.
    
    Args:
        ground_truth_path: Path to data.txt file
    
    Returns:
        List of ground truth name strings (one per line)
    
    Raises:
        FileNotFoundError: If ground truth file doesn't exist
    """
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    return lines


def move_folder_to_category(source_folder: Path, detected: bool, base_dir: Path):
    """
    Move folder to DETECTED or NOT DETECTED category.
    
    Args:
        source_folder: Path to folder to move
        detected: True if name was detected, False otherwise
        base_dir: Base directory (Test_Results)
    """
    try:
        if detected:
            target_dir = base_dir / "DETECTED"
        else:
            target_dir = base_dir / "NOT DETECTED"
        
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_folder.name
        
        # Handle name collision
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{source_folder.name}_{counter}"
            counter += 1
        
        shutil.move(str(source_folder), str(target_path))
        return target_path
    except Exception as e:
        print(f"Warning: Failed to move folder {source_folder} to category: {e}")
        return source_folder


def generate_report(results: List[Dict], output_csv_path: Path, base_results_dir: Path):
    """
    Generate CSV report and print terminal output with mismatches.
    Moves folders to DETECTED or NOT DETECTED based on name extraction.
    
    Args:
        results: List of result dictionaries
        output_csv_path: Path to save CSV file
        base_results_dir: Base results directory for moving folders
    """
    # Write CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['pdf_file', 'output_folder', 'extracted_name', 'found_in_data_txt', 
                     'matched_line', 'strict_match', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Calculate statistics
    total = len(results)
    matched = sum(1 for r in results if r['found_in_data_txt'])
    not_found = sum(1 for r in results if not r['found_in_data_txt'] and not r['error'])
    extraction_failed = sum(1 for r in results if r['error'])
    strict_matches = sum(1 for r in results if r.get('strict_match', False))
    
    # Count detected vs not detected names
    names_detected = sum(1 for r in results if r['extracted_name'] and not r['error'])
    names_not_detected = sum(1 for r in results if not r['extracted_name'] or r['error'])
    
    # Move folders to DETECTED or NOT DETECTED
    print("\nMoving folders to DETECTED / NOT DETECTED...")
    for r in results:
        folder_path = Path(r['output_folder'])
        if folder_path.exists():
            name_detected = r['extracted_name'] and not r['error']
            new_path = move_folder_to_category(folder_path, name_detected, base_results_dir)
            r['output_folder'] = str(new_path)  # Update path in result
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print(f"Total PDFs processed: {total}")
    print(f"\nName Detection:")
    print(f"  ✓ Names detected: {names_detected}")
    print(f"  ✗ Names NOT detected: {names_not_detected}")
    print(f"\nValidation against data.txt:")
    print(f"  ✓ Matched in data.txt: {matched} ({strict_matches} strict matches)")
    print(f"  ✗ NOT found in data.txt: {not_found}")
    print(f"  ⚠ Extraction failed: {extraction_failed}")
    print(f"\nCSV report saved to: {output_csv_path}")
    
    # Print names NOT found in data.txt
    not_found_list = [r for r in results if not r['found_in_data_txt'] and not r['error']]
    extraction_failed_list = [r for r in results if r['error']]
    
    if not_found_list or extraction_failed_list:
        print("\n" + "=" * 80)
        print("Names NOT found in data.txt:")
        print("=" * 80)
        
        for r in not_found_list:
            print(f"  PDF: {r['pdf_file']}")
            print(f"    Extracted name: {r['extracted_name'] or '(empty)'}")
            print(f"    Output folder: {r['output_folder']}")
            print()
        
        if extraction_failed_list:
            print("\nExtraction failures:")
            for r in extraction_failed_list:
                print(f"  PDF: {r['pdf_file']}")
                print(f"    Error: {r['error']}")
                print(f"    Output folder: {r['output_folder']}")
                print()
    else:
        print("\n✓ All extracted names were found in data.txt!")
    
    print("=" * 80)


def process_pdf(pdf_path: Path, output_base_dir: Path, ground_truth_lines: List[str], 
                dry_run: bool = False) -> Dict:
    """
    Process a single PDF through the entire pipeline.
    
    Args:
        pdf_path: Path to PDF file
        output_base_dir: Base directory for outputs
        ground_truth_lines: List of ground truth names
        dry_run: If True, only show what would be done
    
    Returns:
        Result dictionary with processing results
    """
    pdf_basename = pdf_path.stem
    
    # Create unique per-PDF folder
    pdf_output_dir = output_base_dir / pdf_basename
    pdf_output_dir = ensure_unique_folder(pdf_output_dir)
    
    result = {
        'pdf_file': pdf_path.name,
        'output_folder': str(pdf_output_dir),
        'extracted_name': None,
        'found_in_data_txt': False,
        'matched_line': None,
        'strict_match': False,
        'error': None
    }
    
    if dry_run:
        print(f"[DRY RUN] Would process: {pdf_path.name}")
        print(f"  Output folder: {pdf_output_dir}")
        return result
    
    try:
        # Step 1: PDF to PNG
        print(f"Processing {pdf_path.name}...")
        png_paths = pdf_to_png(str(pdf_path), str(pdf_output_dir))
        if not png_paths:
            result['error'] = "PDF to PNG conversion failed"
            return result
        png_path = png_paths[0]  # First page only
        
        # Step 2: Image optimization
        png_basename = Path(png_path).stem
        artifacts = process_image_for_ocr(png_path, str(pdf_output_dir), png_basename)
        
        nombre_roi_path = artifacts['nombre_roi']
        
        # Step 3: Process rectangles and extract name
        rectangles_output = pdf_output_dir / f"rectangles_{Path(nombre_roi_path).name}"
        # Review image if name extraction might fail (we'll check nombre_detected flag)
        rectangles_result = process_rectangles(nombre_roi_path, str(rectangles_output), show_window=False, review_image=False)
        
        # Get extracted name from rectangles (name is extracted between NOMBRE and DOMICILIO, or from start to DOMICILIO)
        extracted_name = rectangles_result.get('extracted_name')
        nombre_detected = rectangles_result.get('nombre_detected', False)
        result['extracted_name'] = extracted_name
        
        # Always print extracted name to console
        print(f"  ✓ Completed: {pdf_path.name}")
        if extracted_name:
            print(f"    Extracted name: {extracted_name}")
            if not nombre_detected:
                print(f"    ⚠ Warning: 'Nombre' marker not detected - extracted from start to Domicilio")
            # Step 4: Match against ground truth
            found, matched_line, strict_match = match_name(extracted_name, ground_truth_lines)
            result['found_in_data_txt'] = found
            result['matched_line'] = matched_line
            result['strict_match'] = strict_match
            
            if result['found_in_data_txt']:
                print(f"    ✓ Matched in data.txt: {result['matched_line']}")
            else:
                print(f"    ✗ NOT found in data.txt")
        else:
            result['error'] = "OCR extraction returned no name"
            print(f"    ✗ Extraction failed: No name extracted")
        
    except Exception as e:
        error_msg = str(e)
        result['error'] = error_msg
        print(f"  ✗ Error processing {pdf_path.name}: {error_msg}")
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch OCR test pipeline")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be processed without executing')
    args = parser.parse_args()
    
    # Validate input directory
    if not PDF_INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {PDF_INPUT_DIR}")
        sys.exit(1)
    
    # Load ground truth
    try:
        ground_truth_lines = load_ground_truth(GROUND_TRUTH_FILE)
        print(f"Loaded {len(ground_truth_lines)} ground truth names from {GROUND_TRUTH_FILE}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Find all PDF files
    pdf_files = sorted(PDF_INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {PDF_INPUT_DIR}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be processed]")
    
    # Process each PDF
    results = []
    for pdf_path in pdf_files:
        result = process_pdf(pdf_path, TEST_RESULTS_DIR, ground_truth_lines, args.dry_run)
        results.append(result)
    
    # Generate report (skip if dry run)
    if not args.dry_run:
        generate_report(results, SUMMARY_CSV, TEST_RESULTS_DIR)
    else:
        print(f"\n[DRY RUN] Would generate report at: {SUMMARY_CSV}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
