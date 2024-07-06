import cv2
import os
import numpy as np


def crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # No face detected

    # Assuming only one face per image, get the first face
    x, y, w, h = faces[0]
    cropped_face = image[y:y + h, x:x + w]

    return cropped_face


def process_image(image_path, save_path, image_name):
    # Load the image
    image = cv2.imread(image_path)

    # Crop to the face region
    cropped_face = crop_face(image)
    if cropped_face is None:
        print(f"No face detected in {image_name}, skipping...")
        return

    # Example resizing to a fixed size (e.g., 224x224)
    resized_image = cv2.resize(cropped_face, (224, 224))

    # Save the final processed image
    cv2.imwrite(os.path.join(save_path, f"{image_name}_cropped.jpg"), resized_image)


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
emotion_images_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\img"

# Path to the folder where processed images will be saved
save_folder = r"C:\Users\ASUS\PycharmProjects\CSC566GroupProject\src\preprocessed_img"

# Create the save folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Process images in each emotion folder
for emotion in os.listdir(emotion_images_folder):
    emotion_path = os.path.join(emotion_images_folder, emotion)
    if os.path.isdir(emotion_path):
        save_path = os.path.join(save_folder, emotion)
        os.makedirs(save_path, exist_ok=True)
        process_images_in_folder(emotion_path, save_path)
