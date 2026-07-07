#!/usr/bin/env python3
"""
Download Amazon Reviews 2023 dataset helper script.

Usage:
    python scripts/download_data.py --sample-size 500
    python scripts/download_data.py --sample-size null  # Full
"""

import sys
import argparse
from pathlib import Path

# Add ml module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.download_dataset import download_dataset


def main():
    parser = argparse.ArgumentParser(description='Download Amazon dataset')
    parser.add_argument(
        '--sample-size',
        type=lambda x: int(x) if x.lower() != 'null' else None,
        default=500,
        help='Sample size (default: 500)'
    )
    parser.add_argument(
        '--category',
        default='Electronics',
        help='Product category (default: Electronics)'
    )
    parser.add_argument(
        '--output-dir',
        default='data',
        help='Output directory (default: data/)'
    )
    
    args = parser.parse_args()
    
    try:
        filepath = download_dataset(
            output_dir=args.output_dir,
            category=args.category,
            sample_size=args.sample_size,
        )
        print(f"\n✅ Success! File ready at: {filepath}")
        print(f"\n📋 Next step:")
        print(f"python scripts/generate_embeddings.py --input {filepath}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
