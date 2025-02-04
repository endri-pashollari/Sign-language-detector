import mediapipe as mp
import pickle
import numpy as np
import os
import cv2

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.p')

with open(MODEL_PATH, 'rb') as model_file:
    model_dict = pickle.load(model_file)
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Cannot access the webcam")
    exit()

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands  
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Class Labels
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

while True:
    data_aux = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,  # Image to draw
                hand_landmarks,  # Model output
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmarks
            x_ = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
            y_ = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize X-coordinate
                data_aux.append(y - min(y_))  # Normalize Y-coordinate

        
        if len(data_aux) == len(x_) * 2:  # Ensure it matches training shape
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            print("Predicted Character:", predicted_character)

            # Display prediction on frame
            cv2.putText(frame, f'Prediction: {predicted_character}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            print("Error: Feature vector size mismatch")
l
    # Show output
    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
