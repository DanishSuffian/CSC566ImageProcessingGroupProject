import cv2
import os
import numpy as np


def assess_image_quality(image):
    # Example: Assess noise level (can be more sophisticated based on your needs)
    mean_intensity = np.mean(image)
    if mean_intensity < 50:
        return 'low_noise'
    elif mean_intensity > 200:
        return 'high_noise'
    else:
        return 'medium_noise'


def apply_noise_reduction(image):
    # Example: Apply noise reduction based on assessed noise level
    noise_level = assess_image_quality(image)
    if noise_level == 'high_noise':
        return cv2.medianBlur(image, 5)
    elif noise_level == 'medium_noise':
        return cv2.medianBlur(image, 3)
    else:
        return image


def apply_contrast_enhancement(image):
    # Example: Always apply contrast enhancement for demonstration
    return cv2.equalizeHist(image)


def apply_sharpening(image):
    # Example: Apply sharpening based on image content
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    if laplacian_var < 100:  # Adjust threshold based on your image characteristics
        return cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    else:
        return image

def apply_smoothing(image):
    # Example: Apply Gaussian Blur for additional smoothing
    return cv2.GaussianBlur(image, (5, 5), 0)

def process_image(image_path, save_path, image_name):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Assess image quality
    noise_level = assess_image_quality(gray_image)

    # Apply appropriate processing based on assessed quality
    if noise_level == 'high_noise':
        processed_image = apply_noise_reduction(gray_image)
    else:
        processed_image = gray_image  # No noise reduction needed for low to medium noise

    # Apply contrast enhancement only for low quality images
    if noise_level == 'low_noise':
        processed_image = apply_contrast_enhancement(processed_image)

    processed_image = apply_sharpening(processed_image)

    processed_image = apply_smoothing(processed_image)

    # Save the final processed image
    cv2.imwrite(os.path.join(save_path, f"{image_name}_processed.jpg"), processed_image)


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
