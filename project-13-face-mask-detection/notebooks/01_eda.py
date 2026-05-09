"""
01_eda.py — Exploratory Data Analysis for the Mask Detection Dataset

Generates and saves all EDA figures to reports/figures/
Run: python notebooks/01_eda.py --data_dir data
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.data_utils import (
    get_class_distribution, load_sample_images, IMG_SIZE,
    get_data_generators,
)
from utils.viz_utils import (
    plot_class_distribution, plot_sample_grid,
    plot_augmentation_preview,
)


def run_eda(data_dir: str = 'data', img_size: int = IMG_SIZE):
    print("\n" + "=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)
    os.makedirs('reports/figures', exist_ok=True)

    # ── 1. Class Distribution ─────────────────────────────────
    print("\n[1] Class distribution...")
    dist = get_class_distribution(data_dir)
    print(f"    {dist}")
    plot_class_distribution(dist)

    total = sum(dist.values())
    for cls, n in dist.items():
        print(f"    {cls:10s}: {n:5d} images ({n/total:.1%})")

    # ── 2. Sample Images ──────────────────────────────────────
    print("\n[2] Sample images per class...")
    samples = load_sample_images(data_dir, n_per_class=5, img_size=img_size)
    plot_sample_grid(samples)

    # ── 3. Image Stats ────────────────────────────────────────
    print("\n[3] Image statistics per class:")
    for cls, imgs in samples.items():
        if not imgs: continue
        arr = np.array(imgs, dtype='float32') / 255.0
        print(f"    [{cls}] mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")

    # ── 4. Augmentation Preview ───────────────────────────────
    print("\n[4] Augmentation preview...")
    cls0 = list(samples.keys())[0]
    if samples[cls0]:
        orig = samples[cls0][0]
        # Simulate augmentations manually
        augs = [
            np.fliplr(orig),
            np.clip((orig * 1.3).astype('uint8'), 0, 255),
            cv2.rotate(orig, cv2.ROTATE_90_CLOCKWISE),
            cv2.GaussianBlur(orig, (5,5), 0),
            cv2.resize(cv2.resize(orig, (80,80)), (img_size, img_size)),
        ]
        plot_augmentation_preview(orig, augs)

    # ── 5. Pixel Distribution ─────────────────────────────────
    print("\n[5] Pixel intensity distributions per class...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Pixel Intensity Distribution', fontsize=13, fontweight='bold')
    colors = ['#4CAF50', '#F44336']
    channels = ['R', 'G', 'B']

    for ax, (cls, imgs), color in zip(axes, samples.items(), colors):
        if not imgs: continue
        arr = np.array(imgs)
        for c, ch in enumerate(channels):
            ax.hist(arr[:,:,:,c].flatten(), bins=50, alpha=0.5,
                    label=f'{ch}', density=True,
                    color=['red','green','blue'][c])
        ax.set_title(f'Class: {cls}', color=color, fontweight='bold')
        ax.set_xlabel('Pixel Value (0-255)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/pixel_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved → reports/figures/pixel_distribution.png")

    # ── 6. Summary ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  EDA COMPLETE")
    print("  Dataset Notes:")
    print("  • CLASSIFICATION ONLY — no bounding boxes")
    print("  • Overcome via pretrained face detection (YOLO)")
    print("  • Classes appear balanced — no resampling needed")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--img_size', type=int, default=IMG_SIZE)
    args = p.parse_args()
    run_eda(args.data_dir, args.img_size)
