"""
Synthetic face data generator for demonstration.
Replace with real Kaggle dataset: 
https://www.kaggle.com/datasets/muhammeddalkran/masked-facerecognition
"""

import numpy as np
import cv2
import os
from pathlib import Path


def draw_face(img, cx, cy, r, skin_color):
    """Draw a simplified face on image."""
    cv2.ellipse(img, (cx, cy), (r, int(r * 1.2)), 0, 0, 360, skin_color, -1)
    # Eyes
    eye_r = max(3, r // 8)
    cv2.circle(img, (cx - r // 3, cy - r // 5), eye_r, (30, 30, 30), -1)
    cv2.circle(img, (cx + r // 3, cy - r // 5), eye_r, (30, 30, 30), -1)
    # Pupils
    cv2.circle(img, (cx - r // 3, cy - r // 5), max(1, eye_r // 2), (0, 0, 0), -1)
    cv2.circle(img, (cx + r // 3, cy - r // 5), max(1, eye_r // 2), (0, 0, 0), -1)
    # Nose
    nose_pts = np.array([
        [cx, cy + r // 6],
        [cx - r // 8, cy + r // 3],
        [cx + r // 8, cy + r // 3]
    ], np.int32)
    cv2.polylines(img, [nose_pts], True, (int(skin_color[0]*0.8), int(skin_color[1]*0.8), int(skin_color[2]*0.8)), 1)
    return img


def draw_mask(img, cx, cy, r, mask_color):
    """Draw a face mask on the lower half of face."""
    mask_pts = np.array([
        [cx - int(r * 0.9), cy + r // 6],
        [cx + int(r * 0.9), cy + r // 6],
        [cx + r, cy + int(r * 1.1)],
        [cx - r, cy + int(r * 1.1)]
    ], np.int32)
    cv2.fillPoly(img, [mask_pts], mask_color)
    # Mask lines (texture)
    for i in range(3):
        y = cy + r // 6 + i * (r // 4)
        cv2.line(img, (cx - int(r * 0.85), y), (cx + int(r * 0.85), y),
                 tuple(max(0, c - 20) for c in mask_color), 1)
    # Ear loops
    cv2.line(img, (cx - int(r * 0.9), cy + r // 6),
             (cx - int(r * 1.05), cy - r // 5), (200, 200, 200), 2)
    cv2.line(img, (cx + int(r * 0.9), cy + r // 6),
             (cx + int(r * 1.05), cy - r // 5), (200, 200, 200), 2)
    return img


def draw_mouth(img, cx, cy, r, skin_color):
    """Draw a mouth (no mask)."""
    mouth_color = (120, 60, 80)
    cv2.ellipse(img, (cx, cy + int(r * 0.55)), (r // 3, r // 6), 0, 0, 180, mouth_color, -1)
    # Lips
    cv2.ellipse(img, (cx, cy + int(r * 0.5)), (r // 3, r // 10), 0, 0, 180, (160, 80, 100), 2)
    return img


def generate_face_image(has_mask: bool, img_size: int = 128, seed: int = None) -> np.ndarray:
    """Generate a synthetic face image with or without a mask."""
    if seed is not None:
        np.random.seed(seed)

    # Random background
    bg_color = tuple(int(x) for x in np.random.randint(150, 230, 3))
    img = np.ones((img_size, img_size, 3), dtype=np.uint8)
    img[:] = bg_color

    # Add subtle gradient background
    for i in range(img_size):
        factor = i / img_size * 30
        img[i, :] = tuple(min(255, max(0, int(c - factor))) for c in bg_color)

    cx, cy = img_size // 2, img_size // 2
    r = img_size // 3

    # Skin tones variety
    skin_tones = [
        (220, 185, 160), (200, 160, 130), (180, 130, 100),
        (160, 110, 80),  (140, 90, 60),   (230, 200, 170)
    ]
    skin_color = skin_tones[np.random.randint(len(skin_tones))]

    # Draw face
    img = draw_face(img, cx, cy, r, skin_color)

    # Hair
    hair_colors = [(30, 20, 10), (60, 40, 20), (100, 80, 60), (180, 140, 80), (40, 40, 40)]
    hair_color = hair_colors[np.random.randint(len(hair_colors))]
    cv2.ellipse(img, (cx, cy - r // 3), (r, int(r * 0.7)), 0, 180, 360, hair_color, -1)

    if has_mask:
        mask_colors = [
            (200, 200, 200), (180, 210, 240), (240, 180, 180),
            (180, 240, 180), (50, 50, 50), (240, 220, 180)
        ]
        mask_color = mask_colors[np.random.randint(len(mask_colors))]
        img = draw_mask(img, cx, cy, r, mask_color)
    else:
        img = draw_mouth(img, cx, cy, r, skin_color)

    # Add noise for realism
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random slight rotation
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img = cv2.warpAffine(img, M, (img_size, img_size),
                         borderMode=cv2.BORDER_REFLECT)

    return img


def generate_dataset(output_dir: str, n_per_class: int = 500, img_size: int = 128):
    """Generate synthetic dataset for mask/no_mask classification."""
    output_dir = Path(output_dir)
    mask_dir = output_dir / "mask"
    no_mask_dir = output_dir / "no_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    no_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_per_class} mask images...")
    for i in range(n_per_class):
        img = generate_face_image(has_mask=True, img_size=img_size, seed=i)
        cv2.imwrite(str(mask_dir / f"mask_{i:04d}.jpg"), img)

    print(f"Generating {n_per_class} no-mask images...")
    for i in range(n_per_class):
        img = generate_face_image(has_mask=False, img_size=img_size, seed=i + 10000)
        cv2.imwrite(str(no_mask_dir / f"no_mask_{i:04d}.jpg"), img)

    print(f"Dataset ready at: {output_dir}")
    print(f"  mask/    : {n_per_class} images")
    print(f"  no_mask/ : {n_per_class} images")
    return str(output_dir)


if __name__ == "__main__":
    generate_dataset("data", n_per_class=600)
