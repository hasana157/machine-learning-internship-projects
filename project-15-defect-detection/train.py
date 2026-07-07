"""
train.py
========
Command-line entry point for training the VisualSentry ConvAutoencoder.

Usage
-----
    python train.py [--config config.yaml]

The script will automatically generate synthetic demo data if the configured
normal image directory is empty, ensuring the project works out-of-the-box
without downloading any external dataset.
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
        description="Train the VisualSentry ConvAutoencoder for defect detection.",
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
    """Run the full training pipeline."""
    args = parse_args()

    if not Path(args.config).exists():
        logger.error("Config file not found: '%s'. Aborting.", args.config)
        sys.exit(1)

    # Import here to allow the CLI to respond quickly before TF loads
    from src.trainer import Trainer

    logger.info("=" * 60)
    logger.info("  VisualSentry — AI-Powered Visual Defect Detection")
    logger.info("  Training Pipeline")
    logger.info("=" * 60)

    trainer = Trainer(config_path=args.config)
    model = trainer.build_model()

    # Print model summary
    model.build_graph()
    model.summary()

    history = trainer.train()

    final_loss = history.history["loss"][-1]
    final_val = history.history.get("val_loss", [None])[-1]
    epochs_run = len(history.history["loss"])

    logger.info("=" * 60)
    logger.info("  Training complete after %d epochs", epochs_run)
    logger.info("  Final train loss : %.6f", final_loss)
    if final_val is not None:
        logger.info("  Final val loss   : %.6f", final_val)
    logger.info("  Model saved to   : %s", trainer.cfg["paths"]["model_save"])
    logger.info("  Loss curve saved : %s", trainer.cfg["paths"]["loss_curve"])
    logger.info("  Training log     : %s", trainer.cfg["paths"]["training_log"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
