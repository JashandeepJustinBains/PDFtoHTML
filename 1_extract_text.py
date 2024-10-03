import easyocr
import cv2
import os

def extract_and_draw_bounding_boxes(image_path, output_path):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        raise ValueError(f"Failed to load the image from {image_path}. Please check the file path and format.")

    # Use EasyOCR to do OCR on the image
    result = reader.readtext(img)

    # Draw bounding boxes on the image
    for (bbox, text, prob) in result:
        # Get the coordinates of the bounding box
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        
        # Draw the rectangle
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        
        # Put the text label
        cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, img)

# Example usage
image_path = 'Image (514).jpg'
output_path = 'output_with_bounding_boxes.jpg'

extract_and_draw_bounding_boxes(image_path, output_path)

print(f"Output image with bounding boxes saved to {output_path}")
