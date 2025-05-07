from keras_facenet import FaceNet
import cv2
import os
import numpy as np

embedder = FaceNet()
known_embeddings = {}
for filename in os.listdir("known_faces"):
    path = os.path.join("known_faces", filename)
    image = cv2.imread(path)
    name = os.path.splitext(filename)[0]
    embedding = embedder.embeddings([image])[0]
    known_embeddings[name] = embedding

np.save("known_embeddings.npy", known_embeddings)
print("Known faces encoded and saved.")
