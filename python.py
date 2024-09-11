import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Directory paths for the dataset
cat_dir = 'C:/Users/mukka/OneDrive/Desktop/3 TASK/cats'  # Folder with cat images
dog_dir = 'C:/Users/mukka/OneDrive/Desktop/3 TASK/dogs'  # Folder with dog images

# Image size to resize all images
IMG_SIZE = 64

def load_images_from_folder(folder, label):
    images = []
    labels = []
    valid_extensions = ['.jpg', '.jpeg', '.png']  # Only process valid image files
    loaded_files = []
    for filename in os.listdir(folder):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img.flatten())  # Flatten the image to a vector
                labels.append(label)
                loaded_files.append(filename)  # Track loaded files
    print(f"Loaded files from {folder}: {loaded_files}")  # Debugging: List the loaded files
    return images, labels

# Load cats and dogs images
cat_images, cat_labels = load_images_from_folder(cat_dir, label=0)  # 0 for cat
dog_images, dog_labels = load_images_from_folder(dog_dir, label=1)  # 1 for dog

# Debugging: Check how many images were loaded
print(f"Number of cat images loaded: {len(cat_images)}")
print(f"Number of dog images loaded: {len(dog_images)}")

# Check if any images were loaded
if len(cat_images) == 0 or len(dog_images) == 0:
    print("Error: One of the directories is empty or contains invalid image files. Please check the directories.")
else:
    # Combine cat and dog data
    X = np.array(cat_images + dog_images)
    y = np.array(cat_labels + dog_labels)

    # Debugging: Check the class distribution
    print(f"Class distribution in the dataset: {np.unique(y, return_counts=True)}")

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Use stratified split to maintain class balance in both training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Check the class distribution in training data
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(f"Error: The training set contains only one class: {unique_classes}. SVM needs more than one class to train.")
    else:
        # Train the Support Vector Machine
        svm = SVC(kernel='linear')  # Linear kernel for simplicity
        svm.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = svm.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Generate classification report
        class_report = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
        print("\nClassification Report:\n", class_report)

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:\n", conf_matrix)
