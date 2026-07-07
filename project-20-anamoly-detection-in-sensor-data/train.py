"""
CLI script to trigger the training pipeline.
"""

from src.utils import load_config, setup_logger
from src.trainer import run_training_pipeline

logger = setup_logger(__name__)

def main():
    try:
        config = load_config("config.yaml")
        _, _ = run_training_pipeline(config)
        print("\n✅ SentinelFlow model trained successfully.\n")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print("\n❌ Training failed. Check logs for details.\n")

if __name__ == "__main__":
    main()
