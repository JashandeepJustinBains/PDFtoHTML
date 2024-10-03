import os
import numpy as np
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import EarlyStopping

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

K.set_image_data_format('channels_last')

# Function to load dataset from pickle files
def load_dataset_from_pickle(data_dir):
    images = []
    labels = []
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'rb') as f:
                class_images, class_labels = pickle.load(f)
                images.extend(class_images)
                labels.extend(class_labels)
                print(f'Loaded labels from {file}: {np.unique(class_labels)}')  # Print unique labels in each file
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=str)  # Ensure labels are read as strings
    return images, labels

# Function to load new dataset
def load_new_dataset(data_dir, label_file):
    images = []
    labels = []
    with open(label_file, 'r') as f:
        label_lines = f.readlines()
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label = label_lines[int(folder)].strip()
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'rb') as img_file:
                    img = Image.open(img_file)
                    img = img.resize((28, 28))  # Resize to 28x28 if needed
                    img_array = np.array(img, dtype=np.uint8)
                    images.append(img_array)
                    labels.append(label)
    
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values
    labels = np.array(labels)
    return images, labels

# Load new dataset
new_data_dir = 'CHoiCe-Dataset/CHoiCe-Dataset/V0.3/data-bin'
label_file = 'CHoiCe-Dataset/CHoiCe-Dataset/V0.3/label.txt'
new_images, new_labels = load_new_dataset(new_data_dir, label_file)
print(f'Number of new images loaded: {len(new_images)}')
print(f'Number of new labels loaded: {len(new_labels)}')

# Map labels to integers
unique_labels = np.unique(new_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_mapping[label] for label in new_labels])

# Convert labels to categorical format
num_classes = len(unique_labels)
new_labels_categorical = to_categorical(mapped_labels, num_classes=num_classes)

# Reshape images
new_images = new_images.reshape(new_images.shape[0], 28, 28, 1)

# Load the existing model
model = load_model('my_model.keras')

# Modify the output layer to match the new number of classes
model.pop()  # Remove the existing output layer
model.add(Dense(num_classes, activation='softmax', name='new_output_layer'))  # Add a new output layer with a unique name

# Recompile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model recompiled with new output layer.")

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fine-tune the model
print("Starting model fine-tuning...")
model.fit(new_images, new_labels_categorical, validation_split=0.2, epochs=10, batch_size=200, callbacks=[early_stopping])
print("Model fine-tuning completed.")

# Save the fine-tuned model
model.save('my_fine_tuned_model.keras')
print("Fine-tuned model saved to 'my_fine_tuned_model.keras'")
