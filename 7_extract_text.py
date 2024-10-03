import os
from google.cloud import vision
from google.cloud.vision import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the path to the service account key file
google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path

def detect_text(image_path):
    client = vision_v1.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    return texts[0].description if texts else ''

def save_text_to_md(text, output_path):
    paragraphs = text.split('\n\n')
    with open(output_path, 'w') as md_file:
        for paragraph in paragraphs:
            if paragraph.strip():
                md_file.write(paragraph.strip() + '\n\n')

if __name__ == "__main__":
    image_path = 'path/to/your/image.png'  # Update this path to your image file
    output_md_path = 'output.md'

    text = detect_text(image_path)
    save_text_to_md(text, output_md_path)

    print(f'Text extracted and saved to {output_md_path}')
