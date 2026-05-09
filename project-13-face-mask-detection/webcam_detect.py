"""
webcam_detect.py — Real-time webcam face mask detection

Opens your webcam and runs the hybrid pipeline on every frame:
  - Haar Cascade / DNN / YOLO detects faces
  - CNN classifier assigns Mask / No Mask + confidence
  - Annotated frames shown live via OpenCV window

Usage:
    python webcam_detect.py                          # uses best available model
    python webcam_detect.py --model models/mask_classifier.keras
    python webcam_detect.py --camera 1               # if you have multiple cameras
    python webcam_detect.py --no_model               # Haar + heuristic only (no CNN)

Press:
    Q  or  ESC  — quit
    S           — save current frame as screenshot
    P           — pause / unpause
"""

import os, sys, time, argparse
import numpy as np
import cv2
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── colors (BGR) ──────────────────────────────────────────────────────────────
GREEN  = (0, 210, 80)
RED    = (40, 40, 220)
YELLOW = (0, 210, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
DARK   = (20, 20, 35)


# ─── Heuristic classifier (no model needed) ──────────────────────────────────
def heuristic_predict(face_bgr: np.ndarray):
    """
    Lightweight heuristic: masks reduce colour saturation in the
    lower-half of the face.  Not accurate — use real CNN when possible.
    """
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV).astype(float)
    mid = face_bgr.shape[0] // 2
    lower_sat = hsv[mid:, :, 1].mean()
    upper_sat = hsv[:mid, :, 1].mean() + 1e-6
    ratio = lower_sat / upper_sat
    has_mask  = ratio < 0.72
    confidence = float(np.clip(abs(ratio - 0.72) * 4 + 0.55, 0.55, 0.95))
    return ("Mask" if has_mask else "No Mask"), confidence


# ─── CNN classifier wrapper ───────────────────────────────────────────────────
class CNNClassifier:
    def __init__(self, model_path: str, img_size: int = 96):
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        print(f"[CNN] Loading model ← {model_path}")
        self.model    = tf.keras.models.load_model(model_path)
        self.img_size = img_size
        # Warm-up pass
        dummy = np.zeros((1, img_size, img_size, 3), dtype="float32")
        self.model.predict(dummy, verbose=0)
        print("[CNN] Model ready.")

    def predict(self, face_bgr: np.ndarray):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_res = cv2.resize(face_rgb, (self.img_size, self.img_size))
        inp      = face_res.astype("float32")[np.newaxis] / 255.0
        score    = float(self.model.predict(inp, verbose=0)[0][0])
        # class_indices: mask=0 (score < 0.5), no_mask=1 (score >= 0.5)
        label = "No Mask" if score >= 0.5 else "Mask"
        conf  = score if score >= 0.5 else 1.0 - score
        return label, conf


# ─── Face detector ────────────────────────────────────────────────────────────
def build_detector(prefer_dnn: bool = True):
    """Try DNN → Haar in order."""
    if prefer_dnn:
        proto = str(ROOT / "models" / "deploy.prototxt")
        caffe = str(ROOT / "models" / "res10_300x300_ssd.caffemodel")
        if os.path.exists(proto) and os.path.exists(caffe):
            net = cv2.dnn.readNet(caffe, proto)
            print("[Det] OpenCV DNN (Res10-SSD) loaded.")
            return "dnn", net

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    print("[Det] Haar Cascade loaded.")
    return "haar", cascade


def detect_faces_dnn(net, frame, conf_thresh=0.50):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                  (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = map(int, box)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2-x1, y2-y1))
    return boxes


def detect_faces_haar(cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40)
    )
    return list(faces) if len(faces) > 0 else []


# ─── Drawing helpers ──────────────────────────────────────────────────────────
def draw_face_box(frame, x, y, w, h, label, confidence, det_time_ms):
    x2, y2 = x + w, y + h
    color   = GREEN if label == "Mask" else RED

    # Thick border box
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    # Filled label background
    text     = f"{label}  {confidence:.0%}"
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.65
    thick    = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    ly = max(y - 10, th + 14)
    cv2.rectangle(frame, (x, ly - th - 10), (x + tw + 12, ly + 4), color, -1)
    cv2.putText(frame, text, (x + 6, ly - 2), font, scale, WHITE, thick, cv2.LINE_AA)

    # Corner brackets for a polished look
    br = 18  # bracket size
    t  = 3
    for (px, py), (dx, dy) in [((x, y), (1, 1)), ((x2, y), (-1, 1)),
                                 ((x, y2), (1,-1)), ((x2,y2),(-1,-1))]:
        cv2.line(frame, (px, py), (px + dx*br, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy*br), color, t)


def draw_hud(frame, fps, n_faces, n_mask, n_nomask, paused, model_type):
    """Top-left HUD overlay."""
    h_frame = frame.shape[0]
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, 160), DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = [
        (f"FPS: {fps:.1f}",                       WHITE,  0.6),
        (f"Model: {model_type}",                   YELLOW, 0.52),
        (f"Faces: {n_faces}",                      WHITE,  0.6),
        (f"Masked:   {n_mask}",                    GREEN,  0.62),
        (f"No Mask: {n_nomask}",                   RED,    0.62),
        ("[ PAUSED ]" if paused else "",           YELLOW, 0.7),
    ]
    y = 26
    for text, color, scale in lines:
        if text:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 2, cv2.LINE_AA)
        y += 24

    # Bottom hint bar
    hint = "Q/ESC: quit   S: screenshot   P: pause"
    cv2.rectangle(frame, (0, h_frame - 28), (frame.shape[1], h_frame), DARK, -1)
    cv2.putText(frame, hint, (10, h_frame - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)


# ─── Main loop ────────────────────────────────────────────────────────────────
def run(
    camera_id:   int  = 0,
    model_path:  str  = None,
    no_model:    bool = False,
    det_conf:    float = 0.45,
    width:       int  = 1280,
    height:      int  = 720,
    save_dir:    str  = "reports/figures/webcam_captures",
):
    # ── classifier ────────────────────────────────────────────────────────────
    if no_model or model_path is None or not os.path.exists(str(model_path)):
        classifier   = None
        model_type   = "Heuristic"
        print("[Mode] No CNN model — using colour-saturation heuristic.")
        print("       (Train the model for real detection accuracy.)")
    else:
        try:
            classifier = CNNClassifier(model_path)
            model_type = "CNN"
        except Exception as e:
            print(f"[Warn] CNN load failed ({e}), falling back to heuristic.")
            classifier = None
            model_type = "Heuristic"

    # ── face detector ─────────────────────────────────────────────────────────
    det_type, detector = build_detector()

    # ── camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera {camera_id}. "
              "Check your camera ID with --camera 0 or --camera 1.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Cam]  Opened camera {camera_id}  ({actual_w}×{actual_h})")
    print("[Ready] Window opening... Press Q or ESC to quit, S to screenshot, P to pause.")

    os.makedirs(save_dir, exist_ok=True)
    paused    = False
    fps_buf   = []
    last_time = time.perf_counter()
    screenshot_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Frame capture failed. Check camera connection.")
                break

        # ── FPS ───────────────────────────────────────────────────────────────
        now = time.perf_counter()
        fps_buf.append(1.0 / max(now - last_time, 1e-6))
        last_time = now
        if len(fps_buf) > 20:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        # ── detect ────────────────────────────────────────────────────────────
        if det_type == "dnn":
            faces = detect_faces_dnn(detector, frame, det_conf)
        else:
            faces = detect_faces_haar(detector, frame)

        n_mask = n_nomask = 0

        t_infer = time.perf_counter()
        for (x, y, fw, fh) in faces:
            x2, y2 = min(x + fw, frame.shape[1]), min(y + fh, frame.shape[0])
            crop = frame[max(0,y):y2, max(0,x):x2]
            if crop.size == 0:
                continue

            if classifier is not None:
                label, conf = classifier.predict(crop)
            else:
                label, conf = heuristic_predict(crop)

            if label == "Mask":
                n_mask   += 1
            else:
                n_nomask += 1

            det_ms = (time.perf_counter() - t_infer) * 1000
            draw_face_box(frame, x, y, fw, fh, label, conf, det_ms)

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(frame, fps, len(faces), n_mask, n_nomask, paused, model_type)

        cv2.imshow("Face Mask Detector  |  Q=quit  S=screenshot  P=pause", frame)

        # ── key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            break
        elif key in (ord('s'), ord('S')):
            screenshot_count += 1
            fname = os.path.join(save_dir, f"capture_{screenshot_count:04d}.jpg")
            cv2.imwrite(fname, frame)
            print(f"[Screenshot] Saved → {fname}")
        elif key in (ord('p'), ord('P')):
            paused = not paused
            print(f"[{'Paused' if paused else 'Resumed'}]")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Saved {screenshot_count} screenshot(s) in {save_dir}/")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time webcam face mask detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--camera",   type=int,   default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--model",    type=str,   default=None,
                        help="Path to trained .keras model\n"
                             "(default: auto-detect models/mask_classifier*.keras)")
    parser.add_argument("--no_model", action="store_true",
                        help="Skip CNN — use heuristic only (no model needed)")
    parser.add_argument("--det_conf", type=float, default=0.45,
                        help="Face detection confidence threshold (default: 0.45)")
    parser.add_argument("--width",    type=int,   default=1280)
    parser.add_argument("--height",   type=int,   default=720)
    args = parser.parse_args()

    # Auto-find model
    model_path = args.model
    if not args.no_model and model_path is None:
        for candidate in [
            str(ROOT / "models" / "mask_classifier_ft.keras"),
            str(ROOT / "models" / "mask_classifier.keras"),
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break

    run(
        camera_id  = args.camera,
        model_path = model_path,
        no_model   = args.no_model,
        det_conf   = args.det_conf,
        width      = args.width,
        height     = args.height,
    )
