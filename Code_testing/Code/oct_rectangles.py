import cv2
import pytesseract
from pytesseract import Output
import os
import re
from typing import Dict, Tuple

myconfig = r"--psm 11 -l spa --oem 3"


def process_rectangles(nombre_roi_path: str, output_path: str, show_window: bool = False, review_image: bool = False) -> Dict[str, str]:
    """
    Process nombre_roi image with pytesseract and draw rectangles around detected text.
    Extracts name between NOMBRE and DOMICILIO. If NOMBRE not found, extracts from start to DOMICILIO.
    
    Args:
        nombre_roi_path: Path to nombre_roi image
        output_path: Path to save the output image with rectangles
        show_window: If True, display the image window (for manual runs). Default False for batch processing.
        review_image: If True, show the image for review even in batch mode. Default False.
    
    Returns:
        Dictionary with:
        - 'output_path': Path to saved output image
        - 'full_ocr_text': Full OCR text (all words)
        - 'extracted_name': Name extracted between NOMBRE and DOMICILIO (or from start to DOMICILIO if NOMBRE not found)
        - 'nombre_detected': Boolean indicating if "Nombre" marker was found
    
    Raises:
        FileNotFoundError: If input image cannot be loaded
    """
    img = cv2.imread(nombre_roi_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image '{nombre_roi_path}'. Please check the file path.")

    # Handle grayscale images (convert to BGR for drawing colored rectangles)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    height, width, _ = img.shape

    data = pytesseract.image_to_data(
        img,
        config=myconfig,
        output_type=Output.DICT
    )

    amount_boxes = len(data['text'])
    
    # Extract all detected text with confidence > 80
    detected_text_parts = []
    full_text_parts = []

    for i in range(amount_boxes):
        text = data['text'][i].strip()
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0
        
        # Store all text for full output
        if text:
            full_text_parts.append(text)
        
        # Process high-confidence text
        if conf > 80:
            detected_text_parts.append(text)
            (x, y, width, height) = (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            )

            img = cv2.rectangle(
                img,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            img = cv2.putText(
                img,
                data['text'][i],
                (x, y + height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
    
    # Get full OCR text and detected text
    full_text = " ".join(full_text_parts)
    detected_text = " ".join(detected_text_parts)
    
    print(f"\n[OCR Results from {os.path.basename(nombre_roi_path)}]")
    print(f"Full OCR text (all words): {full_text}")
    print(f"Detected text (confidence > 80): {detected_text}")
    
    # Extract name between "Nombre" (any combination) and "Domicilio" (any combination)
    # Flexible patterns to match variations: handles OCR errors, spacing, case variations
    # Pattern allows optional spaces between letters to handle OCR splitting
    nombre_variations = [
        r"(?i)n\s*o\s*m\s*b\s*r\s*e\s*:?\s*",  # With optional spaces: n o m b r e
        r"(?i)nombre\s*:?\s*",  # Standard: nombre
        r"(?i)nom\s*bre\s*:?\s*",  # Split: nom bre
        r"(?i)no\s*mbre\s*:?\s*",  # Split: no mbre
    ]
    
    domicilio_variations = [
        r"(?i)d\s*o\s*m\s*i\s*c\s*i\s*l\s*i\s*o\s*:?\s*",  # With optional spaces: d o m i c i l i o
        r"(?i)domicilio\s*:?\s*",  # Standard: domicilio
        r"(?i)domici\s*lio\s*:?\s*",  # Split: domici lio
        r"(?i)dom\s*ici\s*lio\s*:?\s*",  # Split: dom ici lio
        r"(?i)domici\s*:?\s*",  # Abbreviated: domici
        r"(?i)d\s*o\s*m\s*i\s*c\s*i\s*:?\s*",  # Abbreviated with spaces: d o m i c i
    ]
    
    extracted_name = None
    
    # Try to find NOMBRE (any variation) in the full text
    nombre_match = None
    for pattern in nombre_variations:
        nombre_match = re.search(pattern, full_text, re.IGNORECASE)
        if nombre_match:
            break
    
    nombre_detected = False
    
    if nombre_match:
        nombre_detected = True
        text_after_nombre = full_text[nombre_match.end():]
        
        # Look for DOMICILIO (any variation) after NOMBRE
        domicilio_match = None
        for pattern in domicilio_variations:
            domicilio_match = re.search(pattern, text_after_nombre, re.IGNORECASE)
            if domicilio_match:
                break
        
        if domicilio_match:
            # Extract text between NOMBRE and DOMICILIO
            name_text = text_after_nombre[:domicilio_match.start()].strip()
            extracted_name = name_text
            print(f"Extracted name (between Nombre and Domicilio): {extracted_name}")
        else:
            # If no DOMICILIO found, take text after NOMBRE (first reasonable amount)
            name_text = " ".join(text_after_nombre.split()[:15]).strip()
            extracted_name = name_text
            print(f"Extracted name (after Nombre, no Domicilio found): {extracted_name}")
    else:
        # NOMBRE not detected - extract from beginning to DOMICILIO
        print(f"Nombre marker not found - extracting from start to Domicilio")
        
        # Look for DOMICILIO from the beginning
        domicilio_match = None
        for pattern in domicilio_variations:
            domicilio_match = re.search(pattern, full_text, re.IGNORECASE)
            if domicilio_match:
                break
        
        if domicilio_match:
            # Extract text from start to DOMICILIO
            name_text = full_text[:domicilio_match.start()].strip()
            extracted_name = name_text
            print(f"Extracted name (from start to Domicilio): {extracted_name}")
        else:
            # No DOMICILIO found either - use first part of text
            name_text = " ".join(full_text.split()[:15]).strip()
            extracted_name = name_text
            print(f"Extracted name (from start, no Domicilio found): {extracted_name}")
    
    # Save the result image with rectangles
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    cv2.imwrite(output_path, img)
    
    # Show window if explicitly requested (for manual runs) or for review
    if show_window or review_image:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Return result dictionary with full OCR text and extracted name
    result = {
        'output_path': output_path,
        'full_ocr_text': full_text,
        'extracted_name': extracted_name,
        'nombre_detected': nombre_detected
    }
    
    return result


if __name__ == "__main__":
    # Original hardcoded logic for standalone execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, '../Test_Data/Result/nombre_roi_RAFAEL LOPEZ LINARES_page_1.png')
    output_path = os.path.join(script_dir, '../Test_Data/Result', f"rectangles_{os.path.basename(image_path)}")
    
    result = process_rectangles(image_path, output_path, show_window=True)
    print(f"Image loaded: {image_path}")
    print(f"Result saved: {result['output_path']}")
    print(f"Full OCR text: {result['full_ocr_text']}")
    print(f"Extracted name: {result['extracted_name']}")