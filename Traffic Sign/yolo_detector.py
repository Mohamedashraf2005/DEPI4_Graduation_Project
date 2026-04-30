from ultralytics import YOLO

model = YOLO("traffic.pt")

def detect(image):
    results = model(image,conf=0.1)
    return results[0].boxes.xyxy