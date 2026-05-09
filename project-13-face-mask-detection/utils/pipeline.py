"""
pipeline.py — Hybrid Face Mask Detection Pipeline

Two-stage inference:
  Stage 1 → Face Detection (YOLO/DNN/Haar)
  Stage 2 → Mask Classification (CNN)

This is the CORE of the system — it bridges the gap between
the classification-only dataset and real-world detection.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from utils.data_utils import preprocess_face, IMG_SIZE
from utils.viz_utils import annotate_image
from src.face_detector import get_best_detector, FaceDetector, Detection


# ─── Prediction Result ────────────────────────────────────────────────────────
class FacePrediction:
    """Single face detection + classification result."""

    def __init__(
        self,
        box: Tuple[int, int, int, int],
        label: str,
        confidence: float,
        detection_conf: float,
        face_crop: np.ndarray,
    ):
        self.box = box
        self.label = label                    # "Mask" or "No Mask"
        self.confidence = confidence          # Classification confidence
        self.detection_conf = detection_conf  # Face detection confidence
        self.face_crop = face_crop

    def to_dict(self) -> Dict:
        return {
            "box": list(self.box),
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "detection_conf": round(float(self.detection_conf), 4),
        }

    def __repr__(self):
        x1, y1, x2, y2 = self.box
        return (f"FacePrediction(label={self.label!r}, "
                f"conf={self.confidence:.1%}, box=({x1},{y1},{x2},{y2}))")


# ─── Main Pipeline ────────────────────────────────────────────────────────────
class MaskDetectionPipeline:
    """
    End-to-end hybrid pipeline:
    
      Input Image
        ↓ [Stage 1]
      Face Detection (YOLO → DNN → Haar)
        ↓
      Face Crops
        ↓ [Stage 2]
      CNN Classification (Mask / No Mask)
        ↓
      Annotated Output Image
    """

    def __init__(
        self,
        model_path: str = "models/mask_classifier_ft.keras",
        img_size: int = IMG_SIZE,
        conf_threshold: float = 0.5,
        prefer_yolo: bool = True,
    ):
        """
        Args:
            model_path:      Path to trained Keras model.
            img_size:        Input size for classifier.
            conf_threshold:  Minimum classification confidence.
            prefer_yolo:     Prefer YOLO over DNN detector.
        """
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        print("[Pipeline] Loading classifier...")
        self.classifier = tf.keras.models.load_model(model_path)
        print(f"[Pipeline] Classifier loaded ← {model_path}")

        print("[Pipeline] Loading face detector...")
        self.detector = get_best_detector(prefer_yolo)
        print(f"[Pipeline] Detector: {type(self.detector).__name__}")
        print("[Pipeline] Ready.\n")

    def predict_image(
        self,
        image_bgr: np.ndarray,
        det_conf: float = 0.4,
    ) -> Tuple[np.ndarray, List[FacePrediction]]:
        """
        Run full pipeline on a single BGR image.

        Args:
            image_bgr: Input image in BGR format.
            det_conf:  Face detection confidence threshold.

        Returns:
            (annotated_image_bgr, list_of_predictions)
        """
        t0 = time.perf_counter()

        # ── Stage 1: Detect Faces ───────────────────────────────────────────
        detections: List[Detection] = self.detector.detect(image_bgr, det_conf)

        if not detections:
            # No faces found — return image with warning overlay
            out = image_bgr.copy()
            cv2.putText(out, "No faces detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
            return out, []

        # ── Stage 2: Classify Each Face ─────────────────────────────────────
        predictions: List[FacePrediction] = []
        for det in detections:
            face_input = preprocess_face(det.face_crop, self.img_size)
            raw_score = float(self.classifier.predict(face_input, verbose=0)[0][0])

            # class_indices: {"mask": 0, "no_mask": 1} → sigmoid ≥ 0.5 → no_mask
            # Adjust based on actual class_indices mapping
            label = "No Mask" if raw_score >= 0.5 else "Mask"
            confidence = raw_score if raw_score >= 0.5 else 1.0 - raw_score

            predictions.append(FacePrediction(
                box=det.box,
                label=label,
                confidence=confidence,
                detection_conf=det.confidence,
                face_crop=det.face_crop,
            ))

        # ── Stage 3: Annotate ────────────────────────────────────────────────
        det_dicts = [
            {"box": p.box, "label": p.label, "confidence": p.confidence}
            for p in predictions
        ]
        annotated = annotate_image(image_bgr, det_dicts)

        elapsed = time.perf_counter() - t0
        fps_text = f"Inference: {elapsed*1000:.0f}ms | Faces: {len(predictions)}"
        cv2.putText(annotated, fps_text, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return annotated, predictions

    def predict_from_path(self, image_path: str) -> Tuple[np.ndarray, List[FacePrediction]]:
        """Load image from disk and run prediction."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.predict_image(img)

    def predict_batch(
        self,
        image_paths: List[str],
        output_dir: str = "reports/figures/predictions",
    ) -> List[Dict]:
        """
        Batch inference on multiple images.
        Saves annotated outputs to output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        summary = []

        for path in image_paths:
            name = Path(path).stem
            try:
                annotated, preds = self.predict_from_path(path)
                out_path = os.path.join(output_dir, f"{name}_pred.jpg")
                cv2.imwrite(out_path, annotated)
                summary.append({
                    "image": path,
                    "output": out_path,
                    "faces": len(preds),
                    "predictions": [p.to_dict() for p in preds],
                })
                print(f"  ✓ {name} → {len(preds)} face(s) → {out_path}")
            except Exception as e:
                print(f"  ✗ {path}: {e}")
                summary.append({"image": path, "error": str(e)})

        return summary


# ─── Quick Test ───────────────────────────────────────────────────────────────
def run_demo(model_path: str = "models/mask_classifier_ft.keras"):
    """Generate synthetic test images and run the full pipeline."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_generator import generate_face_image

    pipeline = MaskDetectionPipeline(model_path)
    os.makedirs("reports/figures/predictions", exist_ok=True)

    print("\n[Demo] Running on synthetic test images...")
    for i, has_mask in enumerate([True, False, True, False]):
        img_rgb = generate_face_image(has_mask, img_size=256, seed=i * 999)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        annotated, preds = pipeline.predict_image(img_bgr)
        label_true = "mask" if has_mask else "no_mask"
        out_path = f"reports/figures/predictions/demo_{i}_{label_true}.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"  Saved → {out_path}")
        for p in preds:
            print(f"    Face: {p}")

    print("\n[Demo] Complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/mask_classifier_ft.keras")
    parser.add_argument("--image", default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo or args.image is None:
        run_demo(args.model)
    else:
        pipe = MaskDetectionPipeline(args.model)
        annotated, preds = pipe.predict_from_path(args.image)
        cv2.imwrite("output.jpg", annotated)
        print(f"Saved → output.jpg | Detected {len(preds)} face(s)")
        for p in preds:
            print(f"  {p}")
