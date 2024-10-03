import os
import sys
import numpy as np
import cv2
import pickle

# Function to load NIST dataset from train directories
def load_nist_class(data_dir, clas):
    images = []
    labels = []
    class_dir = os.path.join(data_dir, clas)
    for sub_dir in os.listdir(class_dir):
        if sub_dir.startswith('train_'):
            train_dir = os.path.join(class_dir, sub_dir)
            print(f'Loading from folder: {train_dir}')
            files = [f for f in os.listdir(train_dir) if f.endswith('.png')]
            for i, file in enumerate(files):
                # Extract the numeric part of the class (e.g., 4 from 4a)
                label = int(''.join(filter(str.isdigit, clas)))
                img_path = os.path.join(train_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28))
                images.append(img)
                labels.append(label)
                # Print progress
                if (i + 1) % 1000 == 0:
                    print(f'Loaded {i + 1} images so far...')
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    return images, labels

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python 3_load_data.py <class>")
        sys.exit(1)

    clas = sys.argv[1]
    data_dir = 'F:/PDFtoHTML/by_class'
    output_dir = 'F:/PDFtoHTML/by_class_pickle'
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading class {clas}...")
    images, labels = load_nist_class(data_dir, clas)
    print(f'Finished loading class {clas}. Number of images: {len(images)}, Number of labels: {len(labels)}')
    
    # Save class dataset to file
    output_file = os.path.join(output_dir, f'nist_class_{clas}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump((images, labels), f)
    print(f"Class {clas} dataset saved to '{output_file}'")

if __name__ == "__main__":
    main()
