# Sign Language Detector

This project is a machine learning-based **sign language detector** using **Mediapipe** and **Random Forest Classification**.

## Features
- Uses **computer vision** to detect hand gestures.
- Extracts hand **landmarks** with **Mediapipe**.
- Classifies hand signs into **A, B, and C**.
- Real-time detection using OpenCV.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sign-language-detector.git
   cd sign-language-detector
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe scikit-learn numpy
   ```
3. Run the program:
   ```bash
   python inference_classifier.py
   ```

## Training the Model
To retrain the model with new data:
```bash
python train_classifier.py
