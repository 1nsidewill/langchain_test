import os
import fitz  # PyMuPDF
import requests
import io
from PIL import Image
import base64

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Define input and output folders
INPUT_FOLDER = "input_folder"
OUTPUT_FOLDER = "output_folder"

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Function to extract images from a PDF file
def extract_images_from_pdf(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        image_bytes = pix.tobytes()
        image = Image.open(io.BytesIO(image_bytes))
        images.append(image)
    pdf_document.close()
    return images

# Function to use GPT-4 to extract text from an image
def extract_text_from_image(image):
    # Convert PIL image to base64 string for sending to OpenAI
    image_bytes = io.BytesIO()
    if image.mode == 'CMYK':
        image = image.convert('RGB')
    image.save(image_bytes, format='PNG')
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": """
            Extract the text from the following image. 
            Don't improvise, just write down all text from the image. 
            Just answer with texts from the image.
            """
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "high"
            }
            }
        ]
        }
    ],
    "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    
    return response_json['choices'][0]['message']['content'].strip()

# Iterate over files in the input folder
for file_name in os.listdir(INPUT_FOLDER):
    file_path = os.path.join(INPUT_FOLDER, file_name)
    if file_name.lower().endswith(('.pdf', '.png', '.jpeg', '.jpg')):
        all_extracted_text = ""

        if file_name.lower().endswith('.pdf'):
            # Extract images from PDF
            images = extract_images_from_pdf(file_path)
            # Extract text from each image
            for image in images:
                text = extract_text_from_image(image)
                all_extracted_text += text + "\n"
        else:
            # Extract text from image file directly
            image = Image.open(file_path)
            all_extracted_text = extract_text_from_image(image)

        # Write the extracted text to a .txt file in the output folder
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file_name)[0]}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(all_extracted_text)

        # Remove the original file
        os.remove(file_path)

print("Processing complete.")