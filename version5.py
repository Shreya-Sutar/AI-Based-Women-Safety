import cv2
import numpy as np
import sounddevice as sd
from datetime import datetime, timedelta
from collections import deque
import torch
from PIL import Image
import mediapipe as mp
import os
import platform
from twilio.rest import Client
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

torch.set_grad_enabled(False)

# ==============================
# BEEP
# ==============================
def beep(freq=1000, duration=300):
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(freq, duration)

# ==============================
# LOAD MODELS
# ==============================
processor = AutoImageProcessor.from_pretrained(
    "dima806/fairface_gender_image_detection"
)
gender_model = AutoModelForImageClassification.from_pretrained(
    "dima806/fairface_gender_image_detection"
)

weapon_model = YOLO("yolov8n.pt")
WEAPON_CLASSES = ["knife", "scissors"]

# ==============================
# MEDIAPIPE
# ==============================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1, min_detection_confidence=0.7
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==============================
# AUDIO
# ==============================
fs = 16000
SCREAM_THRESHOLD = 0.08
scream_flag = False

def audio_callback(indata, frames, time, status):
    global scream_flag
    volume = np.sqrt(np.mean(indata**2))
    scream_flag = volume > SCREAM_THRESHOLD

audio_stream = sd.InputStream(
    callback=audio_callback, channels=1, samplerate=fs
)
audio_stream.start()

# ==============================
# RISK SCORING ENGINE
# ==============================
RISK_WEIGHTS = {
    "weapon": 5,
    "sos": 4,
    "scream": 2
}

RISK_THRESHOLD = 4
risk_score = 0

# Persistence counters
weapon_frames = 0
sos_frames = 0
scream_frames = 0

PERSISTENCE_REQUIRED = 3

last_alert_time = datetime.min
ALERT_COOLDOWN = timedelta(seconds=20)

# ==============================
# HELPERS
# ==============================
def classify_gender(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(face_rgb)
    inputs = processor(images=img, return_tensors="pt")
    outputs = gender_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    label = gender_model.config.id2label[pred.item()]
    if conf.item() < 0.8:
        label = "Uncertain"
    return label

def detect_sos(results):
    if not results.multi_hand_landmarks:
        return False

    for hand in results.multi_hand_landmarks:
        lm = hand.landmark
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        folded = 0
        for tip, pip in zip(tips, pips):
            if lm[tip].y > lm[pip].y:
                folded += 1

        if folded >= 3:
            return True
    return False

def save_clip(frames):
    if not frames:
        return

    h, w, _ = frames[0].shape
    name = f"alert_{datetime.now().strftime('%H%M%S')}.avi"

    out = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*"XVID"),
        15,
        (w, h)
    )

    for f in frames:
        out.write(f)

    out.release()
    print("Saved:", name)

def send_sms(alert_type, message):
    print(f"[SMS] {alert_type}: {message}")
    # integrate Twilio here

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(2)
buffer = deque(maxlen=150)

print("SYSTEM RUNNING — Press Q to quit")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ==========================
    # GENDER + PEOPLE COUNT
    # ==========================
    male = female = people = 0

    faces = face_detector.process(rgb)
    if faces.detections:
        h, w, _ = frame.shape
        for det in faces.detections:
            box = det.location_data.relative_bounding_box
            x = max(0, int(box.xmin * w))
            y = max(0, int(box.ymin * h))
            bw = int(box.width * w)
            bh = int(box.height * h)

            if bw < 40 or bh < 40:
                continue

            face = frame[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            label = classify_gender(face)

            people += 1
            if label == "Male":
                male += 1
            elif label == "Female":
                female += 1

            cv2.rectangle(frame, (x,y), (x+bw,y+bh), (255,255,255), 2)
            cv2.putText(frame, label, (x,y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ==========================
    # WEAPON DETECTION
    # ==========================
    weapon_found = False
    results = weapon_model(frame, conf=0.35, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            label = weapon_model.names[cls]

            if label in WEAPON_CLASSES:
                weapon_frames += 1
                weapon_found = True
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,label,(x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            else:
                weapon_frames = 0

    # ==========================
    # SOS DETECTION
    # ==========================
    hands_res = hands.process(rgb)
    sos = detect_sos(hands_res)

    if sos:
        sos_frames += 1
    else:
        sos_frames = 0

    # ==========================
    # SCREAM DETECTION
    # ==========================
    if scream_flag:
        scream_frames += 1
    else:
        scream_frames = 0

    # ==========================
    # CALCULATE RISK SCORE
    # ==========================
    risk_score = 0

    if weapon_frames >= PERSISTENCE_REQUIRED:
        risk_score += RISK_WEIGHTS["weapon"]

    if sos_frames >= PERSISTENCE_REQUIRED:
        risk_score += RISK_WEIGHTS["sos"]

    if scream_frames >= PERSISTENCE_REQUIRED:
        risk_score += RISK_WEIGHTS["scream"]

    # ==========================
    # ESCALATION
    # ==========================
    if risk_score >= RISK_THRESHOLD:
        if datetime.now() - last_alert_time > ALERT_COOLDOWN:
            beep(1500, 500)
            send_sms("HIGH RISK", f"Risk Score: {risk_score}")
            save_clip(list(buffer))
            last_alert_time = datetime.now()

    buffer.append(frame.copy())

    # ==========================
    # LEFT SIDE DISPLAY PANEL
    # ==========================
    cv2.putText(frame,f"Males: {male}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,f"Females: {female}",(10,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,f"People: {people}",(10,90),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,f"Risk Score: {risk_score}",(10,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    if sos_frames >= PERSISTENCE_REQUIRED:
        cv2.putText(frame,"SOS DETECTED",(10,160),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)

    if weapon_frames >= PERSISTENCE_REQUIRED:
        cv2.putText(frame,"WEAPON DETECTED",(10,200),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)

    if scream_frames >= PERSISTENCE_REQUIRED:
        cv2.putText(frame,"SCREAM DETECTED",(10,240),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)

    cv2.imshow("Women Safety Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

audio_stream.stop()
audio_stream.close()
cap.release()
cv2.destroyAllWindows()