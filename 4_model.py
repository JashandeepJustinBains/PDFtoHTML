import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import EarlyStopping  # Import EarlyStopping
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

# Load dataset
data_dir = 'F:/PDFtoHTML/by_class_pickle'
print("Loading dataset from pickle files...")
images, labels = load_dataset_from_pickle(data_dir)
print(f'Number of images loaded: {len(images)}')
print(f'Number of labels loaded: {len(labels)}')

# Determine the number of unique classes
unique_labels = np.unique(labels)
num_classes = len(unique_labels)
print(f'Number of unique classes: {num_classes}')
print(f'Unique labels: {unique_labels}')

# Map labels to a range starting from 0
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_mapping[label] for label in labels])

if len(images) == 0 or len(labels) == 0:
    print("Error: No images or labels found. Please check the dataset path and structure.")
else:
    images = images.reshape(images.shape[0], 28, 28, 1).astype('float32') / 255
    labels = to_categorical(mapped_labels, num_classes=num_classes)

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled.")

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train model with EarlyStopping
    print("Starting model training...")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, callbacks=[early_stopping])
    print("Model training completed.")

    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    # Save the model
    model.save('my_model.keras')
    print("Model saved to 'my_model.keras'")

