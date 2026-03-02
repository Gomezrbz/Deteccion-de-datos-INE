from pdf2image import convert_from_path
import os

pdf_path = "../Test_Data/pdf/RAFAEL LOPEZ LINARES.pdf"
output_dir = "../Test_Data/png"

# Extract filename without extension
pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

pages = convert_from_path(
    pdf_path,
    dpi=400,
    fmt="png",
    grayscale=True,
    poppler_path=r"poppler-25.12.0/Library/bin"
)

for i, page in enumerate(pages):
    output_path = os.path.join(output_dir, f"{pdf_filename}_page_{i+1}.png")
    page.save(output_path, "PNG")