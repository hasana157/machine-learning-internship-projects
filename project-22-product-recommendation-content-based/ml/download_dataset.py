#!/usr/bin/env python3
"""
ShopSense AI - Dataset Downloader

Downloads Amazon Reviews 2023 dataset from Hugging Face.

Options:
1. Sample (500 products) - 30 seconds, good for testing
2. Demo (50k products) - 5 minutes, load testing
3. Full (2.5M products) - 2-3 hours, production

Usage:
    python ml/download_dataset.py --sample-size 500
    python ml/download_dataset.py --sample-size null  # Full dataset
    python ml/download_dataset.py --category Electronics --sample-size 10000

Time Complexity:
- Download: O(n) where n = number of products
- Parse: O(n) single pass streaming
- Typical: 500 products in 30s on 10Mbps internet
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Generator
from datetime import datetime

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found.")
    print("Install: pip install datasets huggingface-hub")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Hugging Face dataset identifier
DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"

# Available categories (pick subset for faster download)
AVAILABLE_CATEGORIES = [
    'All_Beauty',
    'Appliances',
    'Arts_Crafts_and_Sewing',
    'Automotive',
    'Baby_Products',
    'Beauty_and_Personal_Care',
    'Books',
    'Camera_and_Photo',
    'Clothing_Shoes_and_Jewelry',
    'Digital_Music',
    'Electronics',
    'Electronics_and_Computers',
    'Furniture',
    'Grocery_and_Gourmet_Food',
    'Health_and_Personal_Care',
    'Home_and_Kitchen',
    'Industrial_and_Scientific',
    'Kindle_Store',
    'Luggage',
    'Musical_Instruments',
    'Office_Products',
    'Pet_Supplies',
    'Shoes',
    'Sports_and_Outdoors',
    'Tools_and_Home_Improvement',
    'Toys_and_Games',
    'Video_Games',
    'Watches',
]

# ============================================================================
# FIELD MAPPING
# ============================================================================

FIELD_MAPPING = {
    # Original → Our field name
    'asin': 'asin',
    'title': 'title',
    'brand': 'brand',
    'description': 'description',
    'features': 'features',
    'categories': 'categories',
    'price': 'price',
    'rating': 'average_rating',
}

# ============================================================================
# DATASET DOWNLOADER
# ============================================================================

def stream_amazon_dataset(
    category: str = 'Electronics',
    sample_size: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Stream Amazon Reviews 2023 dataset from Hugging Face.
    
    Memory-efficient: Generator pattern, not loading all into RAM.
    
    Args:
        category: Category name (e.g., 'Electronics', 'Books')
        sample_size: Limit to N products (None = full category)
    
    Yields:
        Product records (dict)
    """
    
    logger.info(f"Loading Amazon Reviews 2023 - {category} category...")
    
    try:
        # Load dataset from Hugging Face
        # Using streaming=True for memory efficiency
        dataset = load_dataset(
            DATASET_ID,
            category,
            streaming=True,  # Stream instead of download full
            trust_remote_code=True,
        )
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info(f"Available categories: {', '.join(AVAILABLE_CATEGORIES[:5])}...")
        raise
    
    logger.info(f"Streaming products from {category}...")
    
    count = 0
    for record in dataset['full']:
        # Transform fields
        product = {
            'asin': record.get('asin', ''),
            'title': record.get('title', ''),
            'brand': record.get('brand', ''),
            'description': record.get('description', '') or '',
            'features': record.get('features', []) or [],
            'categories': record.get('categories', []) or [],
            'price': float(record.get('price', 0) or 0),
            'average_rating': float(record.get('rating', 0) or 0),
        }
        
        # Validate required fields
        if product['asin'] and product['title']:
            yield product
            count += 1
            
            if sample_size and count >= sample_size:
                logger.info(f"Reached sample size limit: {sample_size}")
                break
            
            if count % 10000 == 0:
                logger.info(f"Streamed {count} products...")
    
    logger.info(f"Total products streamed: {count}")


# ============================================================================
# DATASET WRITER (JSONL)
# ============================================================================

def write_jsonl(
    output_path: str,
    products: Generator[dict, None, None],
    chunk_size: int = 10000,
) -> int:
    """
    Write products to JSONL file in chunks.
    
    JSONL format (one JSON object per line):
        {"asin": "...", "title": "...", ...}
        {"asin": "...", "title": "...", ...}
    
    Args:
        output_path: Output file path
        products: Generator of product records
        chunk_size: Write to disk every N products
    
    Returns:
        Total products written
    """
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing to {output_path}...")
    
    written = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for product in products:
            json_line = json.dumps(product, ensure_ascii=False)
            f.write(json_line + '\n')
            written += 1
            
            # Log progress every chunk
            if written % chunk_size == 0:
                logger.info(f"Written {written} products...")
    
    logger.info(f"✅ Wrote {written} products to {output_path}")
    return written


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def download_dataset(
    output_dir: str = 'data',
    category: str = 'Electronics',
    sample_size: Optional[int] = None,
) -> str:
    """
    Download and prepare Amazon dataset.
    
    Args:
        output_dir: Output directory
        category: Product category
        sample_size: Limit to N products (None = full)
    
    Returns:
        Path to downloaded file
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    size_label = f"_{sample_size}_sample" if sample_size else "_full"
    filename = f"amazon_products{size_label}.jsonl"
    filepath = output_path / filename
    
    logger.info(f"ShopSense AI - Dataset Downloader")
    logger.info(f"=" * 60)
    logger.info(f"Category: {category}")
    logger.info(f"Sample size: {sample_size or 'Full catalogue'}")
    logger.info(f"Output: {filepath}")
    logger.info(f"=" * 60)
    
    start_time = datetime.now()
    
    # Step 1: Stream from Hugging Face
    products = stream_amazon_dataset(category, sample_size)
    
    # Step 2: Write to JSONL
    total_written = write_jsonl(str(filepath), products)
    
    elapsed = datetime.now() - start_time
    
    logger.info(f"=" * 60)
    logger.info(f"✅ Download complete!")
    logger.info(f"Products: {total_written}")
    logger.info(f"Time: {elapsed.total_seconds():.1f}s")
    logger.info(f"File: {filepath}")
    logger.info(f"Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"=" * 60)
    
    # Next steps
    logger.info(f"\n📋 Next steps:")
    logger.info(f"1. Generate ML artefacts:")
    logger.info(f"   python ml/data_ingestion.py --jsonl {filepath} --output ml/artefacts")
    logger.info(f"\n2. Start services:")
    logger.info(f"   docker compose up --build")
    logger.info(f"\n3. Open dashboard:")
    logger.info(f"   http://localhost:8501")
    
    return str(filepath)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface."""
    
    parser = argparse.ArgumentParser(
        description='Download Amazon Reviews 2023 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 500 products (testing, 30 sec)
  python download_dataset.py --sample-size 500
  
  # Download 50k products (load testing, 5 min)
  python download_dataset.py --sample-size 50000
  
  # Download full Electronics category (~500k products, 1-2 hours)
  python download_dataset.py --sample-size null
  
  # Download Books category with 10k samples
  python download_dataset.py --category Books --sample-size 10000
        """
    )
    
    parser.add_argument(
        '--category',
        default='Electronics',
        choices=AVAILABLE_CATEGORIES,
        help='Product category (default: Electronics)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=lambda x: int(x) if x.lower() != 'null' else None,
        default=500,
        help='Sample size (default: 500, "null" for full category)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data',
        help='Output directory (default: data/)'
    )
    
    args = parser.parse_args()
    
    # Download
    try:
        filepath = download_dataset(
            output_dir=args.output_dir,
            category=args.category,
            sample_size=args.sample_size,
        )
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Download cancelled by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
