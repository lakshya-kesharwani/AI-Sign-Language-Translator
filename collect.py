import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

label = input("Enter gesture name: ").replace(" ","_")
os.makedirs(f"data/{label}", exist_ok=True)

count = 0

while True:
    success, img = cap.read()

    if not success:
        print("Camera not working")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for lm in handLms.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])

            np.save(f"data/{label}/{count}.npy", lm_list)
            count += 1

            cv2.putText(img, f"Collecting {label}: {count}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Data Collection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()