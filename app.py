from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import pyttsx3

app = Flask(__name__)

# ---------------- VOICE ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- STATE ----------------
frame_count = 0
prediction = "None"
sentence = ""
last_word = ""
last_time = 0

# ---------------- FRAME GENERATOR ----------------
def generate_frames():
    global frame_count, prediction, sentence, last_word, last_time

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (800, 600))

        frame_count += 1

        # FPS improve
        if frame_count % 5 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    lm_list = []
                    for lm in handLms.landmark:
                        lm_list.extend([lm.x, lm.y, lm.z])

                    lm_list = np.array(lm_list).reshape(1, -1)
                    prediction = model.predict(lm_list)[0]
            else:
                prediction = "None"

        # Sentence build
        current_time = time.time()
        if prediction != "None" and prediction != last_word:
            if current_time - last_time > 2:
                sentence += " " + prediction
                last_word = prediction
                last_time = current_time

        # Display
        cv2.putText(frame, f"Gesture: {prediction}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3)

        cv2.putText(frame, f"Sentence: {sentence}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear')
def clear():
    global sentence
    sentence = ""
    return "cleared"

@app.route('/speak')
def speak():
    global sentence
    if sentence.strip() != "":
        clean_text = sentence.replace("_", " ")
        engine.say(clean_text)
        engine.runAndWait()
    return "spoken"

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)