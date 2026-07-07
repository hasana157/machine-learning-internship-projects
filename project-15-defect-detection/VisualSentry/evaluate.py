"""
evaluate.py
===========
Command-line entry point for evaluating the trained VisualSentry model.

Usage
-----
    python evaluate.py [--config config.yaml]

Loads the saved autoencoder, scores all images in the evaluation set (normal +
defect directories), computes an adaptive threshold from the normal subset, and
outputs classification metrics and a per-image results CSV.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the trained VisualSentry anomaly detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full evaluation pipeline."""
    args = parse_args()

    if not Path(args.config).exists():
        logger.error("Config file not found: '%s'. Aborting.", args.config)
        sys.exit(1)

    import tensorflow as tf
    from src.data_loader import load_config, build_eval_dataset, build_train_dataset
    from src.evaluator import AnomalyEvaluator

    cfg = load_config(args.config)
    model_path = cfg["paths"]["model_save"]

    if not Path(model_path).exists():
        logger.error(
            "No trained model found at '%s'. Run 'make train' or 'python train.py' first.",
            model_path,
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  VisualSentry — AI-Powered Visual Defect Detection")
    logger.info("  Evaluation Pipeline")
    logger.info("=" * 60)

    logger.info("Loading model from '%s' …", model_path)
    model = tf.keras.models.load_model(model_path)

    evaluator = AnomalyEvaluator(model=model, config_path=args.config)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    img_size = tuple(model_cfg["img_size"])
    batch_size = train_cfg["batch_size"]

    # ── Fit threshold on normal images ────────────────────────────────────────
    logger.info("Fitting anomaly threshold on normal training data …")
    normal_dir = paths_cfg["normal_data"]

    normal_images = [str(p) for p in Path(normal_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not normal_images:
        logger.error("No normal images found in '%s'. Run 'make data' first.", normal_dir)
        sys.exit(1)

    def _parse(fp):
        raw = tf.io.read_file(fp)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([img_size[0], img_size[1], 3])
        return img

    normal_ds = (
        tf.data.Dataset.from_tensor_slices(normal_images)
        .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    evaluator.fit_threshold(normal_ds)

    # ── Evaluate on full eval set ──────────────────────────────────────────────
    logger.info("Building evaluation dataset …")
    eval_ds, eval_paths, eval_labels = build_eval_dataset(
        normal_dir=paths_cfg["normal_data"],
        defect_dir=paths_cfg["defect_data"],
        img_size=img_size,
        batch_size=batch_size,
        channels=model_cfg.get("channels", 3),
    )

    logger.info("Scoring all evaluation images …")
    results_df = evaluator.evaluate(eval_ds, eval_paths, eval_labels, save_results=True)

    # ── Compute and display metrics ───────────────────────────────────────────
    predicted_bin = [1 if p == "fail" else 0 for p in results_df["predicted"]]
    scores = results_df["anomaly_score"].values
    metrics = evaluator.compute_metrics(eval_labels, predicted_bin, scores)

    logger.info("=" * 60)
    logger.info("  Evaluation Results")
    logger.info("  Threshold    : %.6f", evaluator.threshold)
    logger.info("  Precision    : %.4f", metrics["precision"])
    logger.info("  Recall       : %.4f", metrics["recall"])
    logger.info("  F1-Score     : %.4f", metrics["f1"])
    logger.info("  AUC-ROC      : %.4f", metrics["auc_roc"])
    logger.info("  Results CSV  : %s", paths_cfg["evaluation_results"])
    logger.info("=" * 60)

    # Summary counts
    n_pass = (results_df["predicted"] == "pass").sum()
    n_fail = (results_df["predicted"] == "fail").sum()
    logger.info("  Passed: %d | Failed: %d | Total: %d", n_pass, n_fail, len(results_df))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
