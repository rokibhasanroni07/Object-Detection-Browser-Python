from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI(title="VisionAI Object Detection API", version="1.0.0")

# Development origins only. Restrict this list to your deployed frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_NAME = "yolo26n.pt"
model = YOLO(MODEL_NAME)  # First launch may download the model weights.

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = 0.35):
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    data = await file.read()
    if not data or len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413 if data else 400, detail="Image must be non-empty and no larger than 10 MB")
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file")
    results = model.predict(image, imgsz=640, conf=confidence, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            detections.append({
                "label": str(model.names[class_id]),
                "class_id": class_id,
                "confidence": float(box.conf[0].item()),
                "box": [x1, y1, x2, y2],
            })
    return {"image_width": image.width, "image_height": image.height, "count": len(detections), "detections": detections}
