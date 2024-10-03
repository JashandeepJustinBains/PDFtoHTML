import cv2
import easyocr
import tensorflow as tf
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the trained model
model = tf.keras.models.load_model('my_fine_tuned_model.keras')

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Read the image using OpenCV
img = cv2.imread('CHoiCe-Dataset/CHoiCe-Dataset/1.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use EasyOCR to do OCR on the grayscale image
result = reader.readtext(gray)

# Create an HTML file and write the extracted text with formatting
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extracted Text</title>
    <style>
        .text-box {
            position: absolute;
            border: 1px solid red;
        }
    </style>
</head>
<body>
    <div style="position: relative;">
        <img src="Image (514).jpg" alt="Scanned Image" style="width: 100%;">
"""

# Add the extracted text to the HTML content
for (bbox, text, prob) in result:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    left = top_left[0]
    top = top_left[1]
    width = top_right[0] - top_left[0]
    height = bottom_left[1] - top_left[1]
    html_content += f'<div class="text-box" style="left: {left}px; top: {top}px; width: {width}px; height: {height}px;">{text}</div>\n'

html_content += """
    </div>
</body>
</html>
"""

# Save the HTML content to a file
with open('extracted_text.html', 'w', encoding='utf-8') as html_file:
    html_file.write(html_content)
