#!/usr/bin/env python3
"""
Test script for CURP Second Version name extraction.

This script tests the extraction function with the test image file.
"""

import sys
import time
from pathlib import Path
from Curp_Second_Version import extract_name_from_image
from utils import check_tesseract_installed, extract_name_curp_second_version

def test_with_ocr_output():
    """Test the extraction function using the OCR output text file."""
    print("=" * 80)
    print("TEST 1: Testing extraction function with OCR output file")
    print("=" * 80)
    
    start_time = time.perf_counter()
    
    ocr_file = Path("data_testing/PNG/output.txt.txt")
    if not ocr_file.exists():
        print(f"ERROR: OCR output file not found: {ocr_file}")
        return False, 0.0
    
    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_text = f.read()
    
    name = extract_name_curp_second_version(ocr_text)
    expected = "RAUL GOMEZ BERMUDEZ"
    #expected = "DIEGO ARMANDO GUTIERREZ CAMPOS"
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"Expected name: {expected}")
    print(f"Extracted name: {name}")
    print(f"Time elapsed: {duration:.3f} seconds")
    
    if name == expected:
        print("✓ TEST PASSED: Name matches expected result")
        return True, duration
    else:
        print("✗ TEST FAILED: Name does not match expected result")
        return False, duration


def test_with_image():
    """Test the full pipeline with the actual image file."""
    print("\n" + "=" * 80)
    print("TEST 2: Testing full pipeline with image file")
    print("=" * 80)
    
    start_time = time.perf_counter()
    
    if not check_tesseract_installed():
        print("SKIPPED: Tesseract OCR is not installed or not in PATH.")
        print("Install Tesseract to run this test.")
        return None, 0.0
    
    #image_path = Path("data_testing/PNG/CURP_PNG2.png")
    image_path = Path("data_testing/PNG/Diego_CURP.png")
    if not image_path.exists():
        print(f"ERROR: Test image not found: {image_path}")
        return False, 0.0
    
    try:
        print("Extracting name from image...")
        middle_time = time.perf_counter()
        first_duration = middle_time - start_time
        print(f"Time elapsed: {first_duration:.3f} seconds")

        name = extract_name_from_image(str(image_path), "CURP")
        #expected = "RAUL GOMEZ BERMUDEZ"
        expected = "DIEGO ARMANDO GUTIERREZ CAMPOS"
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        print(f"Expected name: {expected}")
        print(f"Extracted name: {name}")
        print(f"Time elapsed: {duration:.3f} seconds")
        
        if name == expected:
            print("✓ TEST PASSED: Name matches expected result")
            return True, duration
        else:
            print("✗ TEST FAILED: Name does not match expected result")
            if name:
                print(f"  Got: '{name}' instead of '{expected}'")
            return False, duration
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"✗ TEST FAILED: Error occurred: {e}")
        print(f"Time elapsed: {duration:.3f} seconds")
        import traceback
        traceback.print_exc()
        return False, duration


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CURP Second Version Extraction - Test Suite")
    print("=" * 80 + "\n")
    
    total_start_time = time.perf_counter()
    results = []
    
    # Test 1: Pattern matching with OCR output
    result1, duration1 = test_with_ocr_output()
    results.append(("Pattern Matching", result1, duration1))
    
    # Test 2: Full pipeline with image
    result2 = test_with_image()
    if result2 is not None:
        result2_status, duration2 = result2
        results.append(("Full Pipeline", result2_status, duration2))
    
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, result, duration in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status} ({duration:.3f}s)")
    
    print(f"\nTotal time: {total_duration:.3f} seconds")
    
    all_passed = all(result for _, result, _ in results if result is not None)
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
