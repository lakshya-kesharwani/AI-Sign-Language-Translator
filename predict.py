import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time

# Voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    clean_text = text.replace("_", " ")
    engine.say(clean_text)
    engine.runAndWait()

# Load model
model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

last_time = 0
delay = 2  # seconds

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for lm in handLms.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])

            lm_list = np.array(lm_list).reshape(1, -1)

            prediction = model.predict(lm_list)[0]

            print("Prediction:", prediction)

            cv2.putText(img, f"Sign: {prediction}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

            current_time = time.time()

            # Speak every 2 seconds (no condition)
            if current_time - last_time > delay:
                speak(prediction)
                last_time = current_time

    cv2.imshow("Prediction", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()