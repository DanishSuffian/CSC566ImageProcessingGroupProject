import cv2
import os
import numpy as np


# Function to perform image augmentation for a single emotion folder
def augment_emotion_images(emotion_folder, save_path, num_augmentations=1):
    # Loop through each image in the emotion folder
    for image_file in os.listdir(emotion_folder):
        if image_file.endswith(".jpg"):  # Assuming images are in JPG format
            image_path = os.path.join(emotion_folder, image_file)
            # Load the image
            image = cv2.imread(image_path)

            # Check if the image was loaded successfully
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            # Get the image name without extension
            image_name = os.path.splitext(image_file)[0]

            # Perform augmentations
            for i in range(num_augmentations):
                # Random rotation
                angle = np.random.randint(-30, 30)  # Rotate between -30 and 30 degrees
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(os.path.join(save_path, f"{image_name}_rotate_{i}.jpg"), rotated_image)

                # Horizontal flip
                flipped_image = cv2.flip(image, 1)  # Flip horizontally
                cv2.imwrite(os.path.join(save_path, f"{image_name}_flip_{i}.jpg"), flipped_image)

                # Add noise (optional)
                noisy_image = image + np.random.normal(0, 25, image.shape).astype(np.uint8)  # Add Gaussian noise
                cv2.imwrite(os.path.join(save_path, f"{image_name}_noise_{i}.jpg"), noisy_image)

                # Adjust brightness
                brightness = np.random.randint(-50, 50)  # Adjust brightness by -50 to 50
                brightness_adjusted = np.clip(image + brightness, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, f"{image_name}_brightness_{i}.jpg"), brightness_adjusted)

                # Adjust color (optional)
                color_shift = np.random.randint(-50, 50, size=3)  # Shift color channels by -50 to 50
                color_adjusted = np.clip(image + color_shift, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, f"{image_name}_color_{i}.jpg"), color_adjusted)


# Path to the folder containing emotion images
emotion_images_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\img"

# Path to the folder where augmented images will be saved
save_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\augmented_img"

# Create the save folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Loop through each emotion folder
for emotion in os.listdir(emotion_images_folder):
    emotion_path = os.path.join(emotion_images_folder, emotion)
    if os.path.isdir(emotion_path):
        save_path = os.path.join(save_folder, emotion)
        os.makedirs(save_path, exist_ok=True)
        augment_emotion_images(emotion_path, save_path)
