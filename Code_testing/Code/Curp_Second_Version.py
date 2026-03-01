#!/usr/bin/env python3
"""
Main configuration file for CURP document processing.

This script processes CURP documents and extracts CURP/Clave and name.
Configure the variables below to specify document type, path, and expected name.
"""

import subprocess
import sys
from pathlib import Path
from utils import pdf_to_png, process_text_file, check_tesseract_installed, manage_output_file

# Configuration variables
file_path = "data_testing/PNG/CURP/CURP_Raul.pdf"  # Path to the image/PDF file (PNG or PDF)
name_expected = "Raul Gomez Bermudez"  # Expected name to be extracted
curp_expected = "GOBR940325HDFMRL03"  # Expected CURP/Clave to be extracted
dob_expected = "25/03/1994"  # Expected date of birth in DD/MM/YYYY format


def main():
    """Main function to process document and display results."""
    print("=" * 80)
    print("CURP Document Processing")
    print("=" * 80)
    print(f"File path: {file_path}")
    print(f"Expected name: {name_expected}")
    print(f"Expected CURP: {curp_expected}")
    print(f"Expected DOB:  {dob_expected}")
    print("=" * 80)
    print()
    
    # Check if tesseract is installed
    if not check_tesseract_installed():
        print("ERROR: Tesseract OCR is not installed or not in PATH.")
        print("Cannot generate output.txt file.")
        return 1
    
    # Determine input file path
    input_path = Path(file_path)
    
    if not input_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return 1
    
    # Check if input is a PDF and convert to PNG if needed
    actual_input_path = input_path
    if input_path.suffix.lower() == '.pdf':
        print(f"Detected PDF file: {file_path}")
        print("Converting PDF to PNG (first page only)...")
        try:
            png_paths = pdf_to_png(str(input_path), dpi=400)
            if png_paths:
                actual_input_path = Path(png_paths[0])  # Use first page
                print(f"✓ Converted PDF to PNG: {actual_input_path}")
            else:
                print("ERROR: PDF conversion failed - no PNG files generated.")
                return 1
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 1
        except Exception as e:
            print(f"ERROR: Failed to convert PDF to PNG: {e}")
            return 1
    else:
        print(f"Processing image file: {file_path}")
    
    # Determine output file path (same directory as input file, named output.txt)
    output_dir = actual_input_path.parent
    output_file = output_dir / "output"
    text_file_path = str(output_dir / "output.txt")
    
    # Generate output.txt using tesseract command
    print("Generating output.txt using tesseract command...")
    try:
        # Run tesseract command
        command = "tesseract" + " " + str(actual_input_path) + " " + str(output_file)
        result_tesseract = subprocess.run(command, shell=True, check=True)
        
        if result_tesseract.returncode == 0:
            print(f"✓ Generated output.txt: {text_file_path}")
        else:
            print(f"⚠ Warning: Tesseract command returned error code {result_tesseract.returncode}")
            if result_tesseract.stderr:
                print(f"  Error: {result_tesseract.stderr}")
            print("  Continuing with processing anyway...")
        
        print()
        
    except subprocess.TimeoutExpired:
        print("⚠ Warning: Tesseract command timed out.")
        return 1
    except FileNotFoundError:
        print("⚠ Warning: Tesseract command not found.")
        return 1
    except Exception as e:
        print(f"ERROR: Error running tesseract command: {e}")
        return 1
    
    # Process the generated text file
    result = process_text_file(text_file_path, name_expected, curp_expected, dob_expected)
    
    # Display results
    if result['error']:
        print(f"ERROR: {result['error']}")
        return 1
    
    # Print extracted values prominently
    print("=" * 80)
    print("EXTRACTED DATA")
    print("=" * 80)
    extracted_curp = result['curp'] or 'Not found'
    extracted_name = result['name'] or 'Not found'
    extracted_dob = result.get('dob') or 'Not found'
    print(f"CURP/Clave: {extracted_curp}")
    print(f"Name:       {extracted_name}")
    print(f"DOB:        {extracted_dob}")
    print("=" * 80)
    print()
    
    print("Results:")
    print(f"  CURP/Clave: {extracted_curp}")
    print(f"  Name: {extracted_name}")
    print(f"  DOB: {extracted_dob}")
    print()
    
    # Display CURP match status
    if result['curp'] and curp_expected:
        if result['curp_match']:
            print("✓ CURP matches expected result")
            print(f"  Expected: {curp_expected}")
            print(f"  Got: {result['curp']}")
        else:
            print("✗ CURP does not match expected result")
            print(f"  Expected: {curp_expected}")
            print(f"  Got: {result['curp']}")
    print()
    
    # Display name match status
    if result['name'] and name_expected:
        if result['name_match']:
            print("✓ Name matches expected result")
            print(f"  Expected: {name_expected}")
            print(f"  Got: {result['name']}")
        else:
            print("✗ Name does not match expected result")
            print(f"  Expected: {name_expected}")
            print(f"  Got: {result['name']}")
    print()
    
    # Display DOB match status
    if result.get('dob') and dob_expected:
        if result.get('dob_match', False):
            print("✓ DOB matches expected result")
            print(f"  Expected: {dob_expected}")
            print(f"  Got: {result['dob']}")
        else:
            print("✗ DOB does not match expected result")
            print(f"  Expected: {dob_expected}")
            print(f"  Got: {result['dob']}")
    print()
    
    # Display timing information
    print("Timing Information:")
    timing = result['timing']
    print(f"  File load time: {timing['file_load']:.3f} seconds")
    if 'ocr' in timing:
        print(f"  OCR time: {timing['ocr']:.3f} seconds")
    print(f"  CURP extraction time: {timing['curp_extraction']:.3f} seconds")
    if 'dob_extraction' in timing:
        print(f"  DOB extraction time: {timing['dob_extraction']:.3f} seconds")
    print(f"  Name extraction time: {timing['name_extraction']:.3f} seconds")
    print(f"  Total processing time: {timing['total']:.3f} seconds")
    print()
    
    # Get match status from result
    curp_match = result.get('curp_match', False)
    name_match = result.get('name_match', False)
    
    # Manage output file (delete if both match, rename if not)
    manage_output_file(str(text_file_path), curp_match, name_match)
    
    if result['success']:
        print("Processing completed successfully!")
        return 0
    else:
        print("Processing completed with issues.")
        return 1


if __name__ == "__main__":
    exit(main())
