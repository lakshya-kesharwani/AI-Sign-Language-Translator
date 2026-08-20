# AI Sign Language Translator

An AI-powered real-time Sign Language Translator that recognizes hand gestures through a webcam and converts them into text and speech.

The project aims to reduce the communication gap between sign-language users and people who do not understand sign language by providing a simple, accessible and real-time translation system.

---

##  Problem Statement

Communication can be challenging for people who use sign language when interacting with people who do not understand it.

Traditional communication methods may require a human interpreter or specialized solutions. This project explores an AI-based approach where hand gestures can be detected through a normal webcam and converted into understandable text and speech.

---

##  Proposed Solution

The system uses computer vision and machine learning to recognize hand gestures from a live camera feed.

The workflow is:

Camera Input
      ↓
Hand Detection
      ↓
21 Hand Landmarks
      ↓
Machine Learning Model
      ↓
Gesture Prediction
      ↓
Sentence Formation
      ↓
Text + Speech Output

---

## ✨ Features

- 🤟 Real-time hand gesture recognition
- 📷 Webcam-based interaction
- 🧠 Machine learning based gesture classification
- 📝 Automatic sentence formation
- 🔊 Text-to-speech output
- 🧹 Clear sentence functionality
- 🌐 Browser-based user interface
- ⚡ Lightweight computer-vision pipeline
- 💻 Designed to run locally

---

## 🛠️ Technologies Used

### Python
Main programming language used to build the application.

### OpenCV
Used for webcam access and real-time video frame processing.

### MediaPipe
Used for real-time hand detection and extraction of 21 hand landmarks.

### NumPy
Used for numerical processing and preparing landmark data for the machine learning model.

### Joblib
Used to load the trained machine learning model stored as `model.pkl`.

### Flask
Used to create the web application and stream the camera feed through the browser.

### Pyttsx3
Used for converting the generated text/sentence into speech.

### Tkinter / PIL
Used in earlier/local versions of the application for the desktop-based interface.

---

## 📂 Project Structure

```text
AI-Sign-Language-Translator/
│
├── app.py
├── collect.py
├── final_app.py
├── old_app.py
├── predict.py
├── train.py
├── model.pkl
│
└── templates/
    └── index.html
