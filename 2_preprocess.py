import csv
import cv2
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

def load_corrected_data(csv_file, image_path):
    images = []
    labels = []
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load the image from {image_path}. Please check the file path and format.")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) != 2:
                print(f"Skipping row with unexpected number of values: {row}")
                continue
            text, bounding_box = row
            bounding_box = eval(bounding_box)  # Convert string to list
            top_left = tuple(map(int, bounding_box[0]))
            bottom_right = tuple(map(int, bounding_box[2]))
            cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cropped_img = cv2.resize(cropped_img, (128, 32))
            cropped_img = cropped_img.astype('float32') / 255.0
            cropped_img = np.expand_dims(cropped_img, axis=-1)
            images.append(cropped_img)
            labels.append(text)
    return np.array(images), np.array(labels)

# Example usage
csv_file = 'handwritten_text.csv'
image_path = 'Image (514).jpg'
images, labels = load_corrected_data(csv_file, image_path)

# Tokenize the labels
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(labels)
sequences = tokenizer.texts_to_sequences(labels)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to numpy array
labels = np.array(padded_sequences)

# Load the pre-trained model
model = load_model('handwriting_recognition_model.h5')

# Add additional layers if needed
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=10, validation_split=0.2)

# Save the fine-tuned model
model.save('fine_tuned_handwriting_recognition_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(images, labels)
print(f'Accuracy: {accuracy}')

# Save the model
model.save('fine_tuned_handwriting_recognition_model.h5')
