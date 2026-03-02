import cv2
import pytesseract
from pytesseract import Output
import os

myconfig = r"--psm 11 -l spa --oem 3"

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Using the nombre_roi image from the Result folder
image_path = os.path.join(script_dir, '../Test_Data/Result/nombre_roi_RAFAEL LOPEZ LINARES_page_1.png')

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image '{image_path}'. Please check the file path.")

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

for i in range(amount_boxes):
    if float(data['conf'][i]) > 80:
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

# Save the result image with rectangles
output_path = os.path.join(script_dir, '../Test_Data/Result', f"rectangles_{os.path.basename(image_path)}")
cv2.imwrite(output_path, img)
print(f"Image loaded: {image_path}")
print(f"Result saved: {output_path}")

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()