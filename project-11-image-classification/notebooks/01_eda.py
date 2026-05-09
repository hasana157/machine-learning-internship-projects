# %% [markdown]
# # CIFAR-10 Exploratory Data Analysis
#
# **Purpose:** Understand the dataset before touching any model.
# This notebook covers:
# 1. Class distribution
# 2. Sample image visualisation
# 3. Pixel intensity distribution per channel
# 4. Augmentation preview
#
# All figures are saved to `reports/figures/`.

# %% [code]
import sys
from pathlib import Path

# Make src/ importable when running from notebooks/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

from src.data_loader import load_raw_cifar10, build_augmentation_layer_baseline
from src.config import CIFAR10_CLASSES, FIGURES_DIR
from utils.visualization import (
    plot_class_distribution,
    plot_sample_images,
    plot_pixel_intensity_distribution,
    plot_augmentation_comparison,
)

# %% [markdown]
# ## 1. Load CIFAR-10

# %% [code]
(x_train, y_train), (x_test, y_test) = load_raw_cifar10()

print(f"Training set : {x_train.shape}  labels: {y_train.shape}")
print(f"Test set     : {x_test.shape}   labels: {y_test.shape}")
print(f"Pixel range  : [{x_train.min()}, {x_train.max()}]")
print(f"Data type    : {x_train.dtype}")

# %% [markdown]
# ## 2. Class Distribution
#
# CIFAR-10 is perfectly balanced: **6 000 training images per class** and
# **1 000 test images per class**.  This means:
# - We do NOT need class-weighted loss.
# - Accuracy is a meaningful metric (no majority-class bias).

# %% [code]
plot_class_distribution(y_train, y_test)

# Quick sanity check
for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
    n = (y_train == cls_idx).sum()
    print(f"  {cls_name:<12} → {n:,} training samples")

# %% [markdown]
# ## 3. Sample Images per Class
#
# Observations:
# - **cat / dog**: body shape, fur colour, and background are very similar → expected source of confusion.
# - **automobile / truck**: both rectangular, metallic → another hard pair.
# - **deer / horse**: silhouettes overlap at 32×32 → model may struggle.
# - Resolution is very low — fine-grained texture details (feathers, fur patterns) are often lost.

# %% [code]
plot_sample_images(x_train, y_train, n_per_class=5)

# %% [markdown]
# ## 4. Pixel Intensity Distribution
#
# The three colour channels have different mean values (R≈125, G≈123, B≈114),
# which is why we apply **channel-wise standardisation** rather than simple
# global scaling to [0, 1].  After normalisation the channels will be zero-centred
# and unit-variance, which speeds up gradient descent.

# %% [code]
plot_pixel_intensity_distribution(x_train)

# %% [markdown]
# ### Channel statistics (training set)

# %% [code]
x_f = x_train.reshape(-1, 3).astype(np.float32) / 255.0
for i, ch in enumerate(["Red", "Green", "Blue"]):
    print(f"  {ch}  mean={x_f[:, i].mean():.4f}  std={x_f[:, i].std():.4f}")

# %% [markdown]
# ## 5. Data Augmentation Preview
#
# We apply the following random transforms **only during training**:
# - Horizontal flip (objects are horizontally symmetric in the real world)
# - Random translation ±10% (shift invariance)
# - Random rotation ±10° (orientation invariance)
# - Random zoom ±10% (scale invariance)
#
# These transforms preserve label semantics (a flipped cat is still a cat)
# while significantly expanding the effective dataset size.

# %% [code]
augment_layer = build_augmentation_layer_baseline()
plot_augmentation_comparison(x_train, augment_layer, n=6)

# %% [markdown]
# ## Summary
#
# | Property             | Value                          |
# |----------------------|-------------------------------|
# | Total images         | 60 000 (50 K train / 10 K test) |
# | Classes              | 10, perfectly balanced         |
# | Image size           | 32 × 32 × 3                   |
# | Hard class pairs     | cat↔dog, automobile↔truck, deer↔horse |
# | Normalisation needed | Yes — channel-wise             |
# | Augmentation needed  | Yes — small dataset, risk of overfitting |

print("\nEDA complete. All figures saved to reports/figures/")
