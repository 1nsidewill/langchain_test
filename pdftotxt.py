import os
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
import pytesseract
from pymilvus import Collection, connections



# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    elements = partition_pdf(filename=pdf_path)
    return " ".join([element.text for element in elements if element.text])

# Function to extract text from an image using OCR
def extract_text_from_image(image_path, lang='kor'):
    text = pytesseract.image_to_string(image_path, lang=lang)
    return text

# Example usage: converting a PDF
pdf_path = 'path_to_your_pdf.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
print("Extracted Text from PDF:", pdf_text)

# Example usage: converting an image
image_path = 'path_to_your_image.png'
image_text = extract_text_from_image(image_path)
print("Extracted Text from Image:", image_text)