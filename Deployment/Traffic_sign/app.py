from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np


app = FastAPI(title="Traffic Sign Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = YOLO("traffic.pt")

@app.post("/predict")
async def predict_traffic(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image"}

    results = model(frame, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "box": [int(x) for x in box.xyxy[0].tolist()]
            })
            
    return {"service": "traffic_sign_detection", "detections": detections}