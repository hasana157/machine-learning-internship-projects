"""src/evaluate.py — Metrics, confusion matrix, feature importance, EDA plots."""
import json, logging
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)

logger = logging.getLogger(__name__)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})
PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706"]

def compute_metrics(y_true, y_pred, class_names):
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro")
    wf1  = f1_score(y_true, y_pred, average="weighted")
    pcf  = f1_score(y_true, y_pred, average=None, labels=class_names)
    rep  = classification_report(y_true, y_pred, target_names=class_names)
    m = {"accuracy": round(acc,4), "macro_f1": round(mf1,4), "weighted_f1": round(wf1,4),
         "per_class_f1": {c: round(float(f),4) for c,f in zip(class_names, pcf)}, "report": rep}
    logger.info(f"\n{'='*55}\nEVALUATION RESULTS\n{'='*55}")
    logger.info(f"  Accuracy    : {acc:.4f}")
    logger.info(f"  Macro F1    : {mf1:.4f}")
    logger.info(f"  Weighted F1 : {wf1:.4f}\n{rep}")
    return m

def save_metrics(metrics, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({k:v for k,v in metrics.items() if k != "report"}, f, indent=2)
    logger.info(f"Metrics saved → {output_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, output_path, normalize=True):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cm_raw  = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i,j]:.2%}\n({cm_raw[i,j]:,})",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if cm_norm[i,j] > thresh else "black")
    ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
    ax.set_ylabel("True", fontsize=13, labelpad=10)
    ax.set_title("Confusion Matrix — News Classification", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Confusion matrix → {output_path}")

def plot_feature_importance(top_features, output_path, top_n=15):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    class_names = list(top_features.keys())
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 8))
    if n == 1: axes = [axes]
    for ax, (cls, feats), color in zip(axes, top_features.items(), PALETTE):
        f = feats[:top_n]; names = [x[0] for x in f][::-1]; vals = [x[1] for x in f][::-1]
        bars = ax.barh(names, vals, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(cls, fontsize=14, fontweight="bold", color=color, pad=10)
        ax.set_xlabel("TF-IDF Coefficient", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=7.5)
    fig.suptitle("Top Discriminative Features per Class", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Feature importance → {output_path}")

def plot_class_distribution(labels, class_names, output_path, title="Class Distribution"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    counts = [Counter(labels).get(c, 0) for c in class_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, counts, color=PALETTE, edgecolor="white")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f"{c:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Category", fontsize=12); ax.set_ylabel("Count", fontsize=12)
    ax.set_ylim(0, max(counts)*1.12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Class distribution → {output_path}")
