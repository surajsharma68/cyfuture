import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from numpy.linalg import norm

embedder = FaceNet()
detector = MTCNN()
known_embeddings = np.load("known_embeddings.npy", allow_pickle=True).item()

def recognize(face_embedding):
    name = "Unknown"
    min_dist = 0.8
    for known_name, known_emb in known_embeddings.items():
        dist = norm(face_embedding - known_emb)
        if dist < min_dist:
            min_dist = dist
            name = known_name
    return name

cap = cv2.VideoCapture(0)
print("[INFO] Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)
        face_crop = frame[y:y+h, x:x+w]
        try:
            embedding = embedder.embeddings([face_crop])[0]
            name = recognize(embedding)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        except:
            continue
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:  # ESC 
        break

cap.release()
cv2.destroyAllWindows()
