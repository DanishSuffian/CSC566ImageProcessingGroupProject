import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns


def edge_based_segmentation(image):
    """
    Perform edge-based segmentation on the image.
    """
    # Convert image to grayscale if not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply edge detection (e.g., Sobel)
    edges = sobel(gray_image)

    # Perform segmentation based on edges (e.g., Felzenszwalb)
    segments = felzenszwalb(edges, scale=100, sigma=0.5, min_size=50)

    # Create segmented image
    segmented_image = np.zeros_like(image, dtype=np.float32)
    for seg_val in np.unique(segments):
        segment_mean = np.mean(image[segments == seg_val], axis=0)
        segmented_image[segments == seg_val] = segment_mean

    return segmented_image


def extract_hog_features(image):
    """
    Extract Histogram of Oriented Gradients (HOG) features from an image.
    """
    resized_image = cv2.resize(image, (196, 196))  # Resize image to 64x64
    features, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(28, 28),
                              cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def extract_lbp_features(image):
    """
    Extract Local Binary Pattern (LBP) features from an image.
    """
    resized_image = cv2.resize(image, (196, 196))  # Resize image to 64x64
    lbp = local_binary_pattern(resized_image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 27),
                             range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def preprocess_images(image_folder):
    """
    Preprocess images in a folder by extracting HOG and LBP features and preparing labels.
    """
    images = []
    labels = []

    for emotion in os.listdir(image_folder):
        emotion_path = os.path.join(image_folder, emotion)
        if os.path.isdir(emotion_path):
            for image_file in os.listdir(emotion_path):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(emotion_path, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = edge_based_segmentation(image)
                    hog_features = extract_hog_features(image)
                    lbp_features = extract_lbp_features(image)
                    combined_features = np.hstack((hog_features, lbp_features))
                    images.append(combined_features)
                    labels.append(emotion)

    return np.array(images), np.array(labels)

# Path to the folder containing emotion images
image_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\processed_img2"

# Preprocess images and extract HOG and LBP features
images, labels = preprocess_images(image_folder)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Prepare data for each emotion
X_train_all, X_test_all, y_train_all, y_test_all = [], [], [], []

for emotion in np.unique(labels):
    emotion_indices = np.where(labels == emotion)
    X_emotion = images[emotion_indices]
    y_emotion = labels_categorical[emotion_indices]

    X_train, X_test, y_train, y_test = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)

    X_train_all.append(X_train)
    X_test_all.append(X_test)
    y_train_all.append(y_train)
    y_test_all.append(y_test)

# Combine all training and testing sets
X_train_all = np.vstack(X_train_all)
X_test_all = np.vstack(X_test_all)
y_train_all = np.vstack(y_train_all)
y_test_all = np.vstack(y_test_all)

# Build a more complex neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_all.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_all, y_train_all, epochs=50, batch_size=32, validation_data=(X_test_all, y_test_all), verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_all, y_test_all)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save("nn_hog_lbp_emotion_model.h5")

# Generate predictions for the test set
y_pred = model.predict(X_test_all)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_all, axis=1)

# Classification report
report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
