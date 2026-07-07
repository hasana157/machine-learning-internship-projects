#!/usr/bin/env python3
"""
Generate ML artefacts (TF-IDF matrix, embeddings, vectorizer).

Usage:
    python scripts/generate_embeddings.py --input data/amazon_products_500_sample.jsonl
"""

import sys
import argparse
from pathlib import Path

# Add ml module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_ingestion import run_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(description='Generate ML artefacts')
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        default='ml/artefacts',
        help='Output directory (default: ml/artefacts)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Limit to N products (optional)'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: File not found: {args.input}")
        sys.exit(1)
    
    try:
        print(f"📊 Generating ML artefacts...")
        print(f"   Input: {args.input}")
        print(f"   Output: {args.output}")
        
        artefacts = run_ingestion_pipeline(
            jsonl_path=args.input,
            output_dir=args.output,
            sample_size=args.sample_size,
        )
        
        print(f"\n✅ Success! Artefacts generated:")
        for name, path in artefacts.items():
            print(f"  ✓ {name}: {path}")
        
        print(f"\n📋 Next step:")
        print(f"docker-compose up --build")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
