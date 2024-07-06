import cv2
import numpy as np
import os

# Function to apply watershed segmentation for emotion detection
def segment_emotion(image_path, save_path):
    image = cv2.imread(image_path)
    original = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply gradient calculation (Sobel derivatives) on entire image
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Combine gradients into magnitude and scale to absolute values
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    # Threshold gradient to obtain markers for watershed
    _, sure_fg = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to enhance markers
    kernel = np.ones((3, 3), np.uint8)
    sure_fg = cv2.dilate(sure_fg, kernel, iterations=3)

    # Marker labelling for watershed algorithm
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Apply watershed algorithm on the entire image
    markers = cv2.watershed(original, markers)

    # Overlay segmented region on original image
    original[markers == -1] = [255, 0, 0]  # Outline region of unknown

    # Save the segmented image with the same name as original
    filename = os.path.basename(image_path)
    save_file = os.path.join(save_path, filename)
    cv2.imwrite(save_file, original)


def process_images_in_folder(emotion_folder, save_folder, limit=1000):
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Loop through each image in the emotion folder
    image_files = [f for f in os.listdir(emotion_folder) if f.endswith(".jpg")]

    for idx, image_file in enumerate(image_files):
        if idx >= limit:
            break
        image_path = os.path.join(emotion_folder, image_file)
        segment_emotion(image_path, save_folder)

# Example usage (replace paths with your own):
emotion_images_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\processed_img"
save_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\segmented_img"

os.makedirs(save_folder, exist_ok=True)

for emotion in os.listdir(emotion_images_folder):
    emotion_path = os.path.join(emotion_images_folder, emotion)
    if os.path.isdir(emotion_path):
        save_path = os.path.join(save_folder, emotion)
        os.makedirs(save_path, exist_ok=True)
        process_images_in_folder(emotion_path, save_path)
