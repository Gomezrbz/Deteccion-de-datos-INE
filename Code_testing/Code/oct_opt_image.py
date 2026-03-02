import cv2
import numpy as np
import os

# ---------------------------------------------------------
# End-to-end: Page (big white canvas) -> crop/warp ID card
#            -> extract + optimize "NOMBRE" ROI for OCR
#
# INPUT : a scanned-page PNG where the INE card is small (top-left)
# OUTPUT:
#   - card_cropped.png          (cropped/warped ID card)
#   - nombre_roi.png            (raw Nombre crop)
#   - nombre_ocr.png            (optimized for OCR)
#   - debug_*                   (helpful tuning images)
# ---------------------------------------------------------

# Paths
input_filename_base = "RAFAEL LOPEZ LINARES_page_1"
input_path = f"../Test_Data/png/{input_filename_base}.png"
output_dir = "../Test_Data/Result"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ---------- helpers ----------
def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def line_kernel_45(length=31, thickness=1):
    k = np.zeros((length, length), dtype=np.uint8)
    # 45° "/" direction
    cv2.line(k, (0, length - 1), (length - 1, 0), 1, thickness)
    return k

# ---------- 0) load ----------
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Could not read: {input_path}")

# Handle alpha channel if present
if img.ndim == 3 and img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- 1) find card on the page ----------
# Mask ink (non-white). If your scans are darker/lighter, tune 245 -> 240 or 250.
_, nonwhite = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

# Close gaps so the card becomes one blob
mask = cv2.morphologyEx(
    nonwhite,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
    iterations=2
)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise RuntimeError("No contours found. Try lowering threshold 245 -> 240.")

c = max(contours, key=cv2.contourArea)

# Try 4-point warp for best geometry; fallback to bounding box
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.02 * peri, True)

card = None
if len(approx) == 4:
    pts = approx.reshape(4, 2).astype("float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    card = cv2.warpPerspective(img, M, (maxW, maxH))
else:
    x, y, w, h = cv2.boundingRect(c)
    pad = 10
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)
    card = img[y:y + h, x:x + w].copy()

card_cropped_path = os.path.join(output_dir, f"card_cropped_{input_filename_base}.png")
cv2.imwrite(card_cropped_path, card)
cv2.imwrite(os.path.join(output_dir, f"debug_page_nonwhite_{input_filename_base}.png"), nonwhite)
cv2.imwrite(os.path.join(output_dir, f"debug_page_mask_{input_filename_base}.png"), mask)

# ---------- 2) crop NOMBRE ROI from the card ----------
# These fractions assume a typical INE layout AFTER the card is cropped/warped.
ch, cw = card.shape[:2]

# If needed, tweak these 4 numbers slightly (they are the only "layout knobs"):
x1, y1 = int(0.15 * cw), int(0.18 * ch)
x2, y2 = int(0.62 * cw), int(0.55 * ch)

nombre_roi = card[y1:y2, x1:x2].copy()
nombre_roi_path = os.path.join(output_dir, f"nombre_roi_{input_filename_base}.png")
cv2.imwrite(nombre_roi_path, nombre_roi)

# ---------- 3) optimize NOMBRE ROI for OCR ----------
g = cv2.cvtColor(nombre_roi, cv2.COLOR_BGR2GRAY)

# Upscale (OCR loves pixels)
g = cv2.resize(g, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

# Local contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
g2 = clahe.apply(g)

# Sharpen (nitidez)
blur = cv2.GaussianBlur(g2, (0, 0), 1.0)
sharp = cv2.addWeighted(g2, 1.5, blur, -0.5, 0)

# Remove diagonal security pattern (directional morphology)
lk = line_kernel_45(length=31, thickness=1)  # try length 41 if diagonal pattern still dominates
diag = cv2.morphologyEx(sharp, cv2.MORPH_OPEN, lk, iterations=1)
no_diag = cv2.subtract(sharp, diag)

# Blackhat to boost dark strokes (text)
bh_k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
blackhat = cv2.morphologyEx(no_diag, cv2.MORPH_BLACKHAT, bh_k)
blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

# Threshold
_, bw = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# OCR-friendly polarity: black text on white background
bw = 255 - bw

# Slight stroke thickening (restore letters if they feel thin)
bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

# Light speckle cleanup
bw = cv2.morphologyEx(
    bw,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    iterations=1
)

nombre_ocr_path = os.path.join(output_dir, f"nombre_ocr_{input_filename_base}.png")
cv2.imwrite(nombre_ocr_path, bw)

# Debug outputs for fast tuning
cv2.imwrite(os.path.join(output_dir, f"debug_nombre_sharp_{input_filename_base}.png"), sharp)
cv2.imwrite(os.path.join(output_dir, f"debug_nombre_diag_{input_filename_base}.png"), diag)
cv2.imwrite(os.path.join(output_dir, f"debug_nombre_no_diag_{input_filename_base}.png"), no_diag)
cv2.imwrite(os.path.join(output_dir, f"debug_nombre_blackhat_{input_filename_base}.png"), blackhat)

print("Saved:")
print(f"- {card_cropped_path}   (cropped/warped ID card)")
print(f"- {nombre_roi_path}     (raw Nombre crop)")
print(f"- {nombre_ocr_path}     (optimized for OCR)")
print(f"- debug_*_{input_filename_base}.png            (tuning images)")
print("\nTuning tips:")
print("1) If ROI is off: tweak x1,y1,x2,y2 fractions.")
print("2) If diagonal background remains: line kernel length 31 -> 41.")
print("3) If letters look thin: dilation iterations 1 -> 2 (watch for merging).")
print("4) If letters merge: reduce dilation to 0 or use a (1,2) kernel.")