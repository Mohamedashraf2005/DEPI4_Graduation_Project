import tensorflow as tf
import numpy as np
import json
import cv2

classifier = tf.keras.models.load_model("damged_not.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)


def preprocess(img):
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img.astype(np.float32), axis=0)
    return img


def predict(img):
    img = preprocess(img)
    preds = classifier.predict(img, verbose=0)

    class_id = np.argmax(preds)
    conf = np.max(preds)

    label = class_names[class_id]

    if conf < 0.6:
        return "Uncertain", conf

    return label, conf