import cv2
import pytesseract
from pytesseract import Output
import os

myconfig = r"--psm 11 -l spa --oem 3"

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'TEST1.png')

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image '{image_path}'. Please check the file path.")

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

cv2.imshow("img", img)
cv2.waitKey(0)
