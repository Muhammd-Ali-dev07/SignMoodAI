import cv2
import mediapipe as mp
from deepface import DeepFace
import threading
import numpy as np
import time
import speech_recognition as sr
import pyautogui

# Setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

mood_label = "Analyzing..."
previous_nose_y = None
nod_threshold = 15
last_mood_time = 0

# Thumbs up gesture
def is_thumbs_up(landmarks):
    return all([
        landmarks.landmark[4].y < landmarks.landmark[3].y,
        landmarks.landmark[8].y > landmarks.landmark[5].y,
        landmarks.landmark[12].y > landmarks.landmark[5].y,
        landmarks.landmark[16].y > landmarks.landmark[5].y,
        landmarks.landmark[20].y > landmarks.landmark[5].y
    ])

# Mood detection using MediaPipe face landmarks
def detect_mood(frame, landmarks):
    global mood_label
    try:
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1 = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
        x2, y2 = min(max(xs) + 20, w), min(max(ys) + 20, h)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            mood_label = "Face not clear"
            return

        # Fix for DeepFace - add detector_backend and handle different return types
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        
        # Handle both list and dictionary return types
        if isinstance(result, list):
            mood_label = result[0]['dominant_emotion'].capitalize()
        else:
            mood_label = result['dominant_emotion'].capitalize()
    except Exception as e:
        mood_label = "Face not clear"
        print("Error:", e)

# Listen for voice commands
def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjusting for ambient noise
        while True:
            try:
                # Increased timeout to 15 seconds
                audio = recognizer.listen(source, timeout=15)  # Increased timeout to 15 seconds
                command = recognizer.recognize_google(audio)
                print(f"Command received: {command}")
                
                # Handle commands
                if "click" in command.lower():
                    print("Performing click...")
                    pyautogui.click(button='left')  # Perform left click when "click" is said
                elif "right click" in command.lower():
                    print("Performing right click...")
                    pyautogui.click(button='right')  # Perform right click when "right click" is said
                elif "scroll up" in command.lower():
                    print("Scrolling up...")
                    pyautogui.scroll(10)  # Perform scroll up
                elif "scroll down" in command.lower():
                    print("Scrolling down...")
                    pyautogui.scroll(-10)  # Perform scroll down
                else:
                    print(f"Command '{command}' not recognized.")
                    
            except sr.WaitTimeoutError:
                print("Listening timeout, waiting for command...")
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
            except Exception as e:
                print(f"Error: {e}")

def start_camera():
    global previous_nose_y, last_mood_time
    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
         mp_face.FaceMesh(refine_landmarks=True) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1280, 900))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            sign_label = ""

            # Hand gestures + move mouse
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * screen_w)
                    y = int(index_tip.y * screen_h)
                    pyautogui.moveTo(x, y)

                    if is_thumbs_up(hand_landmarks):
                        sign_label = "Thumbs Up"

            # Face nod + mood - BUT WITHOUT VISUALIZING THE FACE MESH
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Removed the face mesh visualization line:
                    # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_TESSELATION)

                    # Head nod
                    nose_y = face_landmarks.landmark[1].y * frame.shape[0]
                    if previous_nose_y is not None and (nose_y - previous_nose_y) > nod_threshold:
                        pyautogui.click(button='left')
                    previous_nose_y = nose_y

                    # Mood detection every 1s
                    if time.time() - last_mood_time > 1:
                        # Create a clean copy of the frame for emotion detection
                        clean_frame = frame.copy()
                        threading.Thread(target=detect_mood, args=(clean_frame, face_landmarks)).start()
                        last_mood_time = time.time()
                    break  # Only use first detected face

            # Matrix visual
            h, w, _ = frame.shape
            face_roi = frame[h//2 - 50:h//2 + 50, w//2 - 50:w//2 + 50]
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            matrix = cv2.resize(gray, (10, 10), interpolation=cv2.INTER_AREA)
            vector = matrix.flatten()
            dot_product = np.dot(vector, np.ones(100))

            panel = np.zeros((1000, 500, 3), dtype=np.uint8)

            cv2.putText(panel, "Matrix", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
            for i in range(10):
                for j in range(10):
                    val = matrix[i][j]
                    x_box, y_box = 20 + j * 35, 60 + i * 35
                    cv2.rectangle(panel, (x_box, y_box), (x_box + 35, y_box + 35), (80, 80, 80), 1)
                    cv2.putText(panel, str(val), (x_box + 5, y_box + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (144, 255, 144), 1)

            offset_y = 60 + 10 * 35 + 40
            cv2.putText(panel, "Vector (Flattened)", (20, offset_y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)
            for i, val in enumerate(vector):
                x = 20 + (i % 10) * 45
                y = offset_y + 30 + (i // 10) * 30
                cv2.putText(panel, str(val), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.putText(panel, f"Dot Product: {dot_product:.2f}", (20, 980), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

            if sign_label:
                cv2.putText(frame, f"Sign: {sign_label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if mood_label:
                cv2.putText(frame, f"Mood: {mood_label}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            if panel.shape[0] > frame.shape[0]:
                frame = cv2.copyMakeBorder(frame, 0, panel.shape[0] - frame.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            combined = np.hstack((frame, panel))
            cv2.imshow("SignMood AI - Cursor | Nod | Mood | Matrix", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the command listener in a separate thread
    threading.Thread(target=listen_for_commands, daemon=True).start()
    
    # Start the camera
    start_camera()
