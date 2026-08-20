import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
import time

# ---------------- VOICE ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    clean_text = text.replace("_", " ")
    engine.say(clean_text)
    engine.runAndWait()

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ---------------- STATE ----------------
current_prediction = "None"
sentence = ""
last_word = ""
last_time = 0

# ---------------- UI ----------------
root = tk.Tk()
root.title("AI Sign Language Translator")
root.geometry("900x700")
root.configure(bg="#222")

title = tk.Label(root, text="AI Sign Language Translator",
                 font=("Arial", 24), fg="white", bg="#222")
title.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

gesture_label = tk.Label(root, text="Gesture: None",
                         font=("Arial", 18), fg="cyan", bg="#222")
gesture_label.pack(pady=5)

sentence_label = tk.Label(root, text="Sentence: ",
                          font=("Arial", 20), fg="yellow", bg="#222")
sentence_label.pack(pady=10)

def clear_all():
    global sentence, last_word
    sentence = ""
    last_word = ""
    sentence_label.config(text="Sentence: ")

clear_btn = tk.Button(root, text="🧹 Clear",
                      font=("Arial", 14),
                      command=clear_all,
                      bg="red", fg="white")
clear_btn.pack(pady=5)

# ---------------- MAIN LOOP ----------------
def update_frame():
    global current_prediction, sentence, last_word, last_time

    success, img = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    # ✅ FIXES
    img = cv2.flip(img, 1)  # mirror
    img = cv2.resize(img, (700, 500))  # UI visible + no zoom

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    prediction = "None"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for lm in handLms.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])

            lm_list = np.array(lm_list).reshape(1, -1)
            prediction = model.predict(lm_list)[0]

    current_prediction = prediction
    gesture_label.config(text=f"Gesture: {prediction}")

    # ⏱ add word every 2 sec
    current_time = time.time()
    if prediction != "None" and prediction != last_word:
        if current_time - last_time > 2:
            sentence += " " + prediction
            last_word = prediction
            last_time = current_time
            sentence_label.config(text=f"Sentence: {sentence}")

    # 🎥 show camera
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# 🔊 Speak full sentence
def speak_sentence():
    if sentence.strip() != "":
        speak(sentence)

speak_btn = tk.Button(root, text="🔊 Speak Sentence",
                      font=("Arial", 14),
                      command=speak_sentence,
                      bg="green", fg="white")
speak_btn.pack(pady=5)

# ---------------- RUN ----------------
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()

