import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import mahotas as mh


# Function to extract HOG features from an image
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    return hog_features.flatten()


# Function to extract LBP features from an image
def extract_lbp_features(image):
    radius = 1
    n_points = 8 * radius
    lbp = mh.features.lbp(image, radius, n_points)
    return lbp.ravel()



# Function to process segmented images folder and extract features
def process_segmented_images(root_folder):
    X = []
    y = []

    # Iterate through each emotion folder
    for emotion in os.listdir(root_folder):
        emotion_folder = os.path.join(root_folder, emotion)

        if not os.path.isdir(emotion_folder):
            continue

        # Iterate through each image in the emotion folder
        for filename in os.listdir(emotion_folder):
            if filename.endswith(".jpg"):
                image_path = os.path.join(emotion_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale

                # Extract HOG and LBP features
                hog_features = extract_hog_features(image)
                lbp_features = extract_lbp_features(image)

                # Combine features if necessary
                combined_features = np.concatenate((hog_features, lbp_features))

                # Append features and label to X and y lists
                X.append(combined_features)
                y.append(emotion)  # Use folder name as label

    return np.array(X), np.array(y)


# Example CNN model architecture (modify as per your requirements)
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Example usage:
root_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\segmented_img"

# Process segmented images and extract features
X, y = process_segmented_images(root_folder)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CNN model
input_shape = X_train.shape[1]
num_classes = len(np.unique(y_train))  # Number of unique emotion classes
model = create_cnn_model(input_shape, num_classes)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
