import cv2
import os
import numpy as np


def apply_noise_reduction(image):
    # Gaussian Filtering
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

    # Median Filtering
    median_filtered = cv2.medianBlur(gaussian_filtered, 5)

    return median_filtered


def apply_contrast_enhancement(image):
    # Histogram Equalization
    if len(image.shape) == 2:  # Grayscale image
        hist_equalized = cv2.equalizeHist(image)
    else:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        hist_equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Gamma Correction
    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(hist_equalized, look_up_table)

    return gamma_corrected


def apply_morphological_operations(image):
    # Convert to grayscale if the image is colored
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Erosion and Dilation
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(gray_image, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Opening and Closing
    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


def process_image(image_path, save_path, image_name):
    # Load the image
    image = cv2.imread(image_path)

    # Apply noise reduction
    noise_reduced = apply_noise_reduction(image)

    # Apply contrast enhancement
    contrast_enhanced = apply_contrast_enhancement(noise_reduced)

    # Apply morphological operations
    final_processed = apply_morphological_operations(contrast_enhanced)

    # Save the final processed image
    cv2.imwrite(os.path.join(save_path, f"{image_name}_processed.jpg"), final_processed)


def process_images_in_folder(emotion_folder, save_folder, limit=1000):
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Loop through each image in the emotion folder
    image_files = [f for f in os.listdir(emotion_folder) if f.endswith(".jpg")]

    for idx, image_file in enumerate(image_files):
        if idx >= limit:
            break
        image_path = os.path.join(emotion_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        process_image(image_path, save_folder, image_name)


# Path to the folder containing emotion images
emotion_images_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\augmented_img"

# Path to the folder where processed images will be saved
save_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\processed_img"

# Create the save folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Process images in each emotion folder
for emotion in os.listdir(emotion_images_folder):
    emotion_path = os.path.join(emotion_images_folder, emotion)
    if os.path.isdir(emotion_path):
        save_path = os.path.join(save_folder, emotion)
        os.makedirs(save_path, exist_ok=True)
        process_images_in_folder(emotion_path, save_path)
