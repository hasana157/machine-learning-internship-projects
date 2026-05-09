"""
face_detector.py — Face detection abstraction layer.

Primary:  YOLOv8 pretrained model (ultralytics) — best accuracy
Fallback: OpenCV DNN face detector (Res10 SSD) — no extra deps
Fallback2: Haar Cascade — always available

Why detection is needed:
  The Kaggle dataset provides only classification labels (mask/no_mask).
  There are no bounding boxes. In real-world images, faces can appear
  anywhere, at any scale. A classifier alone can't locate them.
  
  Classification vs Detection:
  ┌─────────────────┬────────────────────────────────────────────┐
  │ Classification  │ Answers "What is in this image?"           │
  │                 │ No spatial info — single label per image   │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Detection       │ Answers "Where in this image?" + "What?"   │
  │                 │ Produces bounding boxes + class labels     │
  └─────────────────┴────────────────────────────────────────────┘
  
  Our hybrid approach: Detection (YOLO/DNN) → locates faces →
  Classification CNN → assigns mask/no-mask per face.
"""

import numpy as np
import cv2
import os
import urllib.request
from pathlib import Path
from typing import List, Tuple


# ─── Detection Result ─────────────────────────────────────────────────────────
class Detection:
    """Single face detection result."""
    __slots__ = ("box", "confidence", "face_crop")

    def __init__(
        self,
        box: Tuple[int, int, int, int],
        confidence: float,
        face_crop: np.ndarray,
    ):
        self.box = box                # (x1, y1, x2, y2)
        self.confidence = confidence  # Detection confidence
        self.face_crop = face_crop    # Cropped BGR face region


# ─── Base Detector ────────────────────────────────────────────────────────────
class FaceDetector:
    """Abstract base class for face detectors."""

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        raise NotImplementedError

    @staticmethod
    def _safe_crop(image: np.ndarray, x1, y1, x2, y2, pad: float = 0.1) -> np.ndarray:
        """Crop with boundary clamping and optional padding."""
        h, w = image.shape[:2]
        pw = int((x2 - x1) * pad)
        ph = int((y2 - y1) * pad)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)
        return image[y1:y2, x1:x2]


# ─── YOLO Detector ────────────────────────────────────────────────────────────
class YOLOFaceDetector(FaceDetector):
    """
    Face detection using YOLOv8n-face (pretrained, not trained from scratch).
    
    This model was pre-trained on WiderFace dataset and detects human faces
    in the wild — multiple faces, different scales, occlusion, angles.
    """

    MODEL_URL = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
    MODEL_PATH = "models/yolov8n-face.pt"

    def __init__(self):
        self.model = None
        self._try_load()

    def _try_load(self):
        try:
            from ultralytics import YOLO
            if not Path(self.MODEL_PATH).exists():
                print(f"[YOLO] Downloading face model → {self.MODEL_PATH}")
                os.makedirs("models", exist_ok=True)
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            self.model = YOLO(self.MODEL_PATH)
            print("[YOLO] YOLOv8-face loaded successfully.")
        except Exception as e:
            print(f"[YOLO] Could not load: {e}")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.4) -> List[Detection]:
        if self.model is None:
            return []
        results = self.model(image_bgr, conf=conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            crop = self._safe_crop(image_bgr, x1, y1, x2, y2)
            if crop.size > 0:
                detections.append(Detection((x1, y1, x2, y2), conf, crop))
        return detections


# ─── OpenCV DNN Detector ──────────────────────────────────────────────────────
class DNNFaceDetector(FaceDetector):
    """
    Res10 SSD face detector (OpenCV DNN module).
    Pretrained on 300-VW dataset. No external downloads needed for
    recent OpenCV builds that bundle the model, otherwise downloads from GitHub.
    """

    PROTO_URL = ("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
                 "face_detector/deploy.prototxt")
    MODEL_URL  = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_"
                  "detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
    PROTO_PATH = "models/deploy.prototxt"
    MODEL_PATH = "models/res10_300x300_ssd.caffemodel"

    def __init__(self):
        self.net = None
        self._try_load()

    def _try_load(self):
        try:
            os.makedirs("models", exist_ok=True)
            if not Path(self.PROTO_PATH).exists():
                urllib.request.urlretrieve(self.PROTO_URL, self.PROTO_PATH)
            if not Path(self.MODEL_PATH).exists():
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            self.net = cv2.dnn.readNet(self.MODEL_PATH, self.PROTO_PATH)
            print("[DNN] OpenCV Res10-SSD face detector loaded.")
        except Exception as e:
            print(f"[DNN] Could not load: {e}")
            self.net = None

    def is_available(self) -> bool:
        return self.net is not None

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        if self.net is None:
            return []
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False,
        )
        self.net.setInput(blob)
        detections_raw = self.net.forward()
        results = []
        for i in range(detections_raw.shape[2]):
            conf = float(detections_raw[0, 0, i, 2])
            if conf < conf_threshold:
                continue
            box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = self._safe_crop(image_bgr, x1, y1, x2, y2)
            if crop.size > 0:
                results.append(Detection((x1, y1, x2, y2), conf, crop))
        return results


# ─── Haar Cascade Fallback ────────────────────────────────────────────────────
class HaarFaceDetector(FaceDetector):
    """Classic Haar Cascade detector — always available in OpenCV."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        print("[Haar] Haar Cascade face detector loaded.")

    def is_available(self) -> bool:
        return True

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        results = []
        for (x, y, fw, fh) in faces:
            x1, y1, x2, y2 = x, y, x + fw, y + fh
            crop = self._safe_crop(image_bgr, x1, y1, x2, y2)
            if crop.size > 0:
                results.append(Detection((x1, y1, x2, y2), 1.0, crop))
        return results


# ─── Auto-selecting Factory ───────────────────────────────────────────────────
def get_best_detector(prefer_yolo: bool = True) -> FaceDetector:
    """
    Return best available face detector in priority order:
    YOLO → DNN → Haar
    """
    if prefer_yolo:
        d = YOLOFaceDetector()
        if d.is_available():
            return d

    d = DNNFaceDetector()
    if d.is_available():
        return d

    print("[Detector] Using Haar Cascade (fallback).")
    return HaarFaceDetector()
