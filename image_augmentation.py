import cv2
import os
import numpy as np
import tensorflow as tf


# Function to perform image augmentation for a single emotion folder
def augment_emotion_images(emotion_folder, save_path, target_count=500):
    # Define the ImageDataGenerator with desired augmentations
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    # Load images from the emotion folder
    images = []
    target_size = (224, 224)  # Define the target size for resizing images
    for image_file in os.listdir(emotion_folder):
        if image_file.endswith(".jpg"):  # Assuming images are in JPG format
            image_path = os.path.join(emotion_folder, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                # Ensure the image is resized properly
                image = cv2.resize(image, target_size)  # Resize image
                images.append(image)
            else:
                print(f"Error loading image: {image_path}")

    if len(images) == 0:
        print(f"No valid images found in {emotion_folder}")
        return

    # Convert list of images to numpy array
    images = np.array(images)

    # Calculate how many new images we need to generate
    current_count = len(images)
    needed_count = target_count - current_count

    # Perform augmentations
    augmented_count = 0
    for x, val in enumerate(
            datagen.flow(images, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='jpg')):
        augmented_count += 1
        if augmented_count >= needed_count:
            break


# Path to the folder containing emotion images
emotion_images_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\preprocessed_img"

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
