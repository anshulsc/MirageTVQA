"""
Run OCR extraction on all rendered table images
Uses the ExtractorFactory to select the appropriate OCR method from config
"""
import json
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from src.configs import text_extraction_config as cfg
from text_extraction.extractor_factory import ExtractorFactory


def load_rendering_metadata(image_path):
    """Load the metadata JSON created during rendering phase."""
    rel_path = image_path.relative_to(cfg.VISUAL_IMAGES_DIR)
    metadata_path = cfg.VISUAL_METADATA_DIR / rel_path.parent / f"{image_path.stem}.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    return None


def infer_source_from_metadata(metadata):
    """Infer source type from metadata or path."""
    if not metadata:
        return 'default'
    
    # Try to get source from table_id naming convention: arxiv_xxx, finqa_xxx, wikisql_xxx
    table_id = metadata.get('source_table_id', '').lower()
    
    if table_id.startswith('arxiv'):
        return 'arxiv'
    elif table_id.startswith('finqa'):
        return 'finqa'
    elif table_id.startswith('wiki'):
        return 'wiki'
    
    return 'default'


def process_single_image(args):
    """
    Worker function for parallel processing.
    Uses ExtractorFactory to create the appropriate extractor based on config.
    
    Args:
        args: Tuple containing (image_path_str,)
        
    Returns:
        Dict with status, image, source, and optional error
    """
    image_path_str, = args
    image_path = Path(image_path_str)
    
    try:
        # Load rendering metadata
        metadata = load_rendering_metadata(image_path)
        source = infer_source_from_metadata(metadata)
        
        # Prepare output paths
        rel_path = image_path.relative_to(cfg.VISUAL_IMAGES_DIR)
        output_table_path = cfg.EXTRACTED_TABLES_DIR / rel_path.parent / f"{image_path.stem}.json"
        output_meta_path = cfg.EXTRACTED_METADATA_DIR / rel_path.parent / f"{image_path.stem}.json"
        
        # Create extractor using factory (automatically selects method from config)
        extractor = ExtractorFactory.create_extractor(
            image_path, 
            source=source, 
            metadata=metadata
        )
        
        # Extract and save
        result = extractor.extract_and_save(output_table_path, output_meta_path)
        
        # Print progress
        status_symbol = {
            'success': '✓',
            'low_quality': '⚠',
            'failed': '✗'
        }.get(result['status'], '?')
        
        print(f"{status_symbol} {image_path.name} ({result['status']}) [PID: {os.getpid()}]")
        
        return result
        
    except Exception as e:
        print(f"✗ {image_path.name} (ERROR: {str(e)}) [PID: {os.getpid()}]")
        return {
            'status': 'failed',
            'image': image_path.name,
            'source': 'unknown',
            'error': str(e)
        }


def main():
    start_time = time.time()
    
    # Determine number of workers
    max_workers = cfg.MAX_WORKERS
    if max_workers is None:
        max_workers = 1
        print("⚠️  WARNING: MAX_WORKERS not set. Defaulting to 1.")
    
    print("=" * 70)
    print("   STARTING PHASE 2: TEXT EXTRACTION")
    print("=" * 70)
    print(f"OCR Method: {ExtractorFactory.get_current_method().upper()}")
    print(f"Using {max_workers} parallel worker(s)")
    
    if max_workers > 1:
        if cfg.OCR_METHOD == 'deepseek':
            print("⚠️  Note: Multiple workers with DeepSeek may cause GPU conflicts")
            print("    Consider setting MAX_WORKERS=1 for GPU-based methods")
    
    # Find all rendered images
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(cfg.VISUAL_IMAGES_DIR.rglob(f"*{ext}"))
    
    print(f"\nFound {len(all_images)} images to process\n")
    
    if len(all_images) == 0:
        print("\n[WARN] No images found. Make sure Phase 1 (rendering) has completed.")
        return
    
    # Prepare tasks (convert Path to string for serialization)
    tasks = [(str(img),) for img in all_images]
    
    # Process in parallel (or sequential if max_workers=1)
    all_results = []
    
    if max_workers == 1:
        # Sequential processing (safer for GPU)
        print("Running in SEQUENTIAL mode\n")
        for task in tasks:
            result = process_single_image(task)
            all_results.append(result)
    else:
        # Parallel processing
        print(f"Running in PARALLEL mode with {max_workers} workers\n")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            all_results = list(executor.map(process_single_image, tasks))
    
    # Calculate statistics
    stats = {
        'total': len(all_results),
        'success': sum(1 for r in all_results if r['status'] == 'success'),
        'low_quality': sum(1 for r in all_results if r['status'] == 'low_quality'),
        'failed': sum(1 for r in all_results if r['status'] == 'failed'),
        'by_source': {}
    }
    
    # Calculate by-source statistics
    for result in all_results:
        source = result.get('source', 'unknown')
        if source not in stats['by_source']:
            stats['by_source'][source] = {'success': 0, 'low_quality': 0, 'failed': 0}
        stats['by_source'][source][result['status']] += 1
    
    # Print final summary
    end_time = time.time()
    print("\n" + "=" * 70)
    print("      EXTRACTION COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"OCR Method Used: {cfg.OCR_METHOD.upper()}")
    
    print(f"\nTotal processed: {stats['total']}")
    print(f"  ✓ Success:      {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"  ⚠  Low quality:  {stats['low_quality']} ({stats['low_quality']/stats['total']*100:.1f}%)")
    print(f"  ✗ Failed:       {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    
    print("\nBy Source:")
    for source, source_stats in sorted(stats['by_source'].items()):
        total = sum(source_stats.values())
        success = source_stats['success']
        if total > 0:
            print(f"  {source.upper():10} : {success}/{total} successful ({success/total*100:.1f}%)")
    
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    if stats['total'] > 0:
        print(f"Average time per image: {(end_time - start_time)/stats['total']:.3f} seconds")
        print(f"Throughput: ~{stats['total']/(end_time - start_time):.2f} images/second")
    print("=" * 70)
    
    # Print warnings if applicable
    if stats['failed'] > stats['success'] / 2:
        print("\n⚠️  WARNING: High failure rate detected!")
        print("   Check error logs and ensure the OCR method is properly configured.")
        print(f"   Current method: {cfg.OCR_METHOD}")


if __name__ == '__main__':
    # For CUDA safety with multiprocessing
    if cfg.OCR_METHOD == 'deepseek':
        mp.set_start_method('spawn', force=True)
    main()