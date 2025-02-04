import os
import mediapipe as mp
import cv2
import pickle
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands  
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)  # Initialize hand detector

# Directory where images are stored
# Ensure this matches the save path in sign_language.py
DATA_DIR = os.path.join(os.path.expanduser("~/Downloads"), "data")

# Lists to store extracted landmark data and labels
data = []
labels = []

# Loop through each folder in the "data" directory (each folder represents a different class)
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)  

    if not os.path.isdir(dir_path):  # ✅ Skip .DS_Store and other non-directory files
        continue

    for img_path in os.listdir(dir_path):  # Loop through each image in the folder
        img_file_path = os.path.join(dir_path, img_path)
        
        if not os.path.isfile(img_file_path):  # ✅ Ensure it's a file (skip invalid entries)
            continue

        x_ = []  # Stores X-coordinates for normalization
        y_ = []  # Stores Y-coordinates for normalization
        data_aux = []  # Stores normalized landmark data for one image

        # Read the image
        img = cv2.imread(img_file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB for Mediapipe

        # Process the image with Mediapipe to detect hands
        results = hands.process(img_rgb)

        # If a hand is detected, extract landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract all X and Y coordinates separately for normalization
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # X-coordinate of landmark
                    y = hand_landmarks.landmark[i].y  # Y-coordinate of landmark
                    x_.append(x)  # Append X to list
                    y_.append(y)  # Append Y to list

                # Normalize coordinates by subtracting the minimum X and Y (ensures position invariance)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize X-coordinate
                    data_aux.append(y - min(y_))  # Normalize Y-coordinate

            # Append the extracted and normalized landmark data to the dataset
            data.append(data_aux)
            labels.append(dir_)  # Store the folder name as the label (class)

        # Display the image with landmarks using Matplotlib
        #plt.figure()  # Create a new figure
       # plt.imshow(img_rgb)  # Show the RGB image
        #plt.show()  # Display the image


# Get the directory where this script is located (ensures the pickle file is saved in the correct place)
SIGN_LANG_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the full path for saving the pickle file inside "Sign_lang"
pickle_file_path = os.path.join(SIGN_LANG_DIR, "data.pickle")

# Save extracted data and labels into a pickle file
with open(pickle_file_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)  # Store data as a dictionary

# Print confirmation message after saving
print(f"Saved data.pickle to: {pickle_file_path}")
