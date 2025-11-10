import json
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

from src.configs import text_extraction_config as cfg
from src.table_ocr.extractor_factory import ExtractorFactory


def infer_source_from_filename(image_path):
    """Infer source type from filename pattern."""
    filename = image_path.name.lower()
    
    # Check filename for source indicators
    if 'arxiv' in filename:
        return 'arxiv'
    elif 'finqa' in filename:
        return 'finqa'
    elif 'wiki' in filename:
        return 'wiki'
    
    # Default to 'default' source
    return 'default'


def is_already_processed(image_path):
    """
    Check if image has already been processed successfully.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if already processed, False otherwise
    """
    try:
        # Prepare output paths
        rel_path = image_path.relative_to(cfg.VISUAL_IMAGES_DIR)
        output_table_path = cfg.EXTRACTED_TABLES_DIR / rel_path.parent / f"{image_path.stem}.json"
        output_meta_path = cfg.EXTRACTED_METADATA_DIR / rel_path.parent / f"{image_path.stem}.json"
        
        # Check if both files exist
        if not output_table_path.exists() or not output_meta_path.exists():
            return False
        
        # Check if metadata indicates successful processing
        with open(output_meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            status = metadata.get('status', 'failed')
            # Consider 'success' and 'low_quality' as already processed
            return status in ['success', 'low_quality']
    
    except Exception as e:
        # If any error occurs, assume not processed
        return False


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
        # Check if already processed
        if is_already_processed(image_path):
            return {
                'status': 'skipped',
                'image': image_path.name,
                'source': infer_source_from_filename(image_path)
            }
        
        # Infer source from filename
        source = infer_source_from_filename(image_path)
        
        # Prepare output paths
        rel_path = image_path.relative_to(cfg.VISUAL_IMAGES_DIR)
        output_table_path = cfg.EXTRACTED_TABLES_DIR / rel_path.parent / f"{image_path.stem}.json"
        output_meta_path = cfg.EXTRACTED_METADATA_DIR / rel_path.parent / f"{image_path.stem}.json"
        
        # Create extractor using factory (no metadata provided)
        extractor = ExtractorFactory.create_extractor(
            image_path, 
            source=source, 
            metadata=None  # No metadata files used
        )
        
        # Extract and save
        result = extractor.extract_and_save(output_table_path, output_meta_path)
        
        return result
        
    except Exception as e:
        return {
            'status': 'failed',
            'image': image_path.name,
            'source': 'unknown',
            'error': str(e)
        }


def get_source_statistics(images):
    """
    Calculate per-source statistics for images.
    
    Args:
        images: List of image paths
        
    Returns:
        Dict with per-source counts
    """
    source_stats = {}
    
    for img in images:
        source = infer_source_from_filename(img)
        if source not in source_stats:
            source_stats[source] = {
                'total': 0,
                'processed': 0,
                'remaining': 0
            }
        
        source_stats[source]['total'] += 1
        
        if is_already_processed(img):
            source_stats[source]['processed'] += 1
        else:
            source_stats[source]['remaining'] += 1
    
    return source_stats


def process_batch(batch_tasks, max_workers, progress_bar):
    """
    Process a batch of images with progress tracking.
    
    Args:
        batch_tasks: List of tasks for this batch
        max_workers: Number of parallel workers
        progress_bar: tqdm progress bar instance
        
    Returns:
        List of results for this batch
    """
    batch_results = []
    
    if max_workers == 1:
        # Sequential processing
        for task in batch_tasks:
            result = process_single_image(task)
            batch_results.append(result)
            
            # Update progress bar with status
            status_symbol = {
                'success': '✓',
                'low_quality': '⚠',
                'failed': '✗',
                'skipped': '⊙'
            }.get(result['status'], '?')
            
            progress_bar.set_postfix_str(
                f"{status_symbol} {result['image'][:40]}",
                refresh=True
            )
            progress_bar.update(1)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(process_single_image, batch_tasks):
                batch_results.append(result)
                
                # Update progress bar with status
                status_symbol = {
                    'success': '✓',
                    'low_quality': '⚠',
                    'failed': '✗',
                    'skipped': '⊙'
                }.get(result['status'], '?')
                
                progress_bar.set_postfix_str(
                    f"{status_symbol} {result['image'][:40]}",
                    refresh=True
                )
                progress_bar.update(1)
    
    return batch_results


def main():
    start_time = time.time()
    
    # Determine number of workers
    max_workers = cfg.MAX_WORKERS
    if max_workers is None:
        max_workers = 1
        print("⚠️  WARNING: MAX_WORKERS not set. Defaulting to 1.")
    
    # Get batch size from config (or use default)
    batch_size = getattr(cfg, 'BATCH_SIZE', 100)
    
    print("=" * 70)
    print("   STARTING PHASE 2: TEXT EXTRACTION (EN PREFIX ONLY)")
    print("=" * 70)
    print(f"OCR Method: {ExtractorFactory.get_current_method().upper()}")
    print(f"Using {max_workers} parallel worker(s)")
    print(f"Batch size: {batch_size} images per batch")
    print(f"Filter: Processing only files with 'en_' prefix")
    print(f"Skip Logic: Enabled (skipping already processed images)")
    print(f"Metadata: Not required (source inferred from filename)")
    print("=" * 70)
    
    if max_workers > 1:
        if cfg.OCR_METHOD == 'deepseek':
            print("⚠️  Note: Multiple workers with DeepSeek may cause GPU conflicts")
            print("    Consider setting MAX_WORKERS=1 for GPU-based methods")
    
    # Find all rendered images
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    all_images = []
    
    print("\n🔍 Scanning for images...")
    for ext in image_extensions:
        all_images.extend(cfg.VISUAL_IMAGES_DIR.rglob(f"*{ext}"))
    
    # FILTER: Only keep images with 'en_' prefix
    filtered_images = [
        img for img in all_images 
        if img.name.startswith('en_')
    ]
    
    print(f"Found {len(all_images)} total images")
    print(f"Filtered to {len(filtered_images)} images with 'en_' prefix")
    
    if len(filtered_images) == 0:
        print("\n[WARN] No 'en_' prefix images found. Make sure Phase 1 (rendering) has completed.")
        print("       and that English language files exist in your dataset.")
        return
    
    # Calculate per-source statistics
    print("\n📊 Analyzing per-source status...")
    source_stats = get_source_statistics(filtered_images)
    
    # Display per-source breakdown
    print("\n" + "=" * 70)
    print("                    PER-SOURCE BREAKDOWN")
    print("=" * 70)
    print(f"{'Source':<15} {'Total':>10} {'Processed':>12} {'Remaining':>12}")
    print("-" * 70)
    
    total_all = 0
    total_processed = 0
    total_remaining = 0
    
    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        total_all += stats['total']
        total_processed += stats['processed']
        total_remaining += stats['remaining']
        
        print(f"{source.upper():<15} {stats['total']:>10} {stats['processed']:>12} {stats['remaining']:>12}")
    
    print("-" * 70)
    print(f"{'TOTAL':<15} {total_all:>10} {total_processed:>12} {total_remaining:>12}")
    print("=" * 70)
    
    # Summary
    print(f"\n📈 Overall Status:")
    print(f"   Total images:      {total_all}")
    print(f"   Already processed: {total_processed} ({total_processed/total_all*100:.1f}%)")
    print(f"   Remaining:         {total_remaining} ({total_remaining/total_all*100:.1f}%)")
    
    if total_remaining == 0:
        print("\n✅ All images have already been processed. Nothing to do!")
        return
    
    # Prepare tasks (convert Path to string for serialization)
    tasks = [(str(img),) for img in filtered_images]
    
    # Split into batches
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    
    print(f"\n🚀 Starting processing of {len(tasks)} images in {total_batches} batch(es)")
    print("=" * 70 + "\n")
    
    # Process batches with overall progress bar
    all_results = []
    
    # Create main progress bar
    with tqdm(
        total=len(tasks),
        desc="Overall Progress",
        unit="img",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ncols=100
    ) as pbar:
        
        for batch_num, batch in enumerate(batches, 1):
            # Update progress bar description with batch info
            pbar.set_description(f"Batch {batch_num}/{total_batches}")
            
            # Process batch
            batch_results = process_batch(batch, max_workers, pbar)
            all_results.extend(batch_results)
    
    # Calculate statistics
    stats = {
        'total': len(all_results),
        'success': sum(1 for r in all_results if r['status'] == 'success'),
        'low_quality': sum(1 for r in all_results if r['status'] == 'low_quality'),
        'failed': sum(1 for r in all_results if r['status'] == 'failed'),
        'skipped': sum(1 for r in all_results if r['status'] == 'skipped'),
        'by_source': {}
    }
    
    # Calculate by-source statistics
    for result in all_results:
        source = result.get('source', 'unknown')
        if source not in stats['by_source']:
            stats['by_source'][source] = {
                'success': 0, 'low_quality': 0, 'failed': 0, 'skipped': 0
            }
        stats['by_source'][source][result['status']] += 1
    
    # Print final summary
    end_time = time.time()
    print("\n" + "=" * 70)
    print("              EXTRACTION COMPLETE - FINAL SUMMARY")
    print("=" * 70)
    print(f"OCR Method Used: {cfg.OCR_METHOD.upper()}")
    print(f"Batch Size: {batch_size} images per batch")
    
    print(f"\n📊 Processing Results:")
    print(f"   Total processed: {stats['total']}")
    print(f"   ✓ Success:       {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"   ⚠ Low quality:   {stats['low_quality']} ({stats['low_quality']/stats['total']*100:.1f}%)")
    print(f"   ✗ Failed:        {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"   ⊙ Skipped:       {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    
    # Calculate actual processing stats (excluding skipped)
    actual_processed = stats['total'] - stats['skipped']
    if actual_processed > 0:
        print(f"\n🆕 Newly Processed (excluding skipped):")
        print(f"   Total new:       {actual_processed}")
        print(f"   ✓ Success:       {stats['success']} ({stats['success']/actual_processed*100:.1f}%)")
        print(f"   ⚠ Low quality:   {stats['low_quality']} ({stats['low_quality']/actual_processed*100:.1f}%)")
        print(f"   ✗ Failed:        {stats['failed']} ({stats['failed']/actual_processed*100:.1f}%)")
    
    print("\n📋 Results by Source:")
    print(f"{'Source':<15} {'Success':>10} {'Low Qual':>10} {'Failed':>10} {'Skipped':>10} {'Total':>10}")
    print("-" * 70)
    for source in sorted(stats['by_source'].keys()):
        source_stats = stats['by_source'][source]
        total = sum(source_stats.values())
        print(f"{source.upper():<15} "
              f"{source_stats['success']:>10} "
              f"{source_stats['low_quality']:>10} "
              f"{source_stats['failed']:>10} "
              f"{source_stats['skipped']:>10} "
              f"{total:>10}")
    
    print("\n⏱️  Performance Metrics:")
    print(f"   Total time:              {end_time - start_time:.2f} seconds")
    if actual_processed > 0:
        print(f"   Time per new image:      {(end_time - start_time)/actual_processed:.3f} seconds")
        print(f"   Throughput (new):        {actual_processed/(end_time - start_time):.2f} images/second")
    if stats['total'] > 0:
        print(f"   Overall throughput:      {stats['total']/(end_time - start_time):.2f} images/second")
    print("=" * 70)
    
    # Print warnings if applicable
    if actual_processed > 0 and stats['failed'] > stats['success'] / 2:
        print("\n⚠️  WARNING: High failure rate detected!")
        print("   Check error logs and ensure the OCR method is properly configured.")
        print(f"   Current method: {cfg.OCR_METHOD}")
    
    if stats['skipped'] > 0:
        print(f"\n✅ Successfully skipped {stats['skipped']} already-processed images")


if __name__ == '__main__':
    if cfg.OCR_METHOD == 'deepseek':
        mp.set_start_method('spawn', force=True)
    main()