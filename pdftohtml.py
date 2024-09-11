import sys
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"F:/Programs/Tesseract-OCR/tesseract.exe"

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=r'F:/poppler/Library/bin')
    return images

def image_to_data(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return data

def detect_color(image, x, y, w, h):
    roi = image[y:y+h, x:x+w]
    avg_color_per_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color[:3]  # Return BGR color

def generate_html(image, data):
    html_content = "<html><head><style>"
    html_content += "body { font-family: Arial, sans-serif; }"
    html_content += ".text { position: absolute; padding: 2px; }"
    html_content += "</style></head><body>"

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:  # Confidence threshold
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text = data['text'][i]
            text_color = detect_color(image, x, y, w, h)
            # text_color_hex = '#%02x%02x%02x' % (int(text_color[2]), int(text_color[1]), int(text_color[0]))  # Convert BGR to HEX
            text_color_hex = '#FFFFFF'
            bg_color = detect_color(image, x, y, w, h)
            bg_color_hex = '#%02x%02x%02x' % (int(bg_color[2]), int(bg_color[1]), int(bg_color[0]))  # Convert BGR to HEX
            style = f"left:{x}px; top:{y}px; width:{w}px; height:{h}px; color:{text_color_hex}; background-color:{bg_color_hex};"
            html_content += f"<div class='text' style='{style}'>{text}</div>"

    html_content += "</body></html>"
    return html_content

def pdf_to_html(pdf_path):
    images = pdf_to_images(pdf_path)
    html_pages = []
    for image in images:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        data = image_to_data(image)
        html = generate_html(image_cv, data)
        html_pages.append(html)
    return html_pages

if __name__ == "__main__":
    if len(sys.argv) == 2:
        pdf_path = sys.argv[1]

        html_pages = pdf_to_html(pdf_path)
        for i, html in enumerate(html_pages):
            with open(f'page_{i+1}.html', 'w') as file:
                file.write(html)
