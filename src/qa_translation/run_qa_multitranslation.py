import json
import time
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, Manager, Lock
from functools import partial
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from src.configs import qa_translation_config as cfg
from src.qa_translation.translator_multi import QATranslator

def process_single_table_wrapper(table_with_id, completed_dict, lock, total_languages):
    """Wrapper to unpack worker_id and table_info"""
    worker_id, table_info = table_with_id
    return process_single_table(table_info, completed_dict, lock, total_languages, worker_id)

def process_single_table(table_info, completed_dict, lock, total_languages, worker_id):
    """Process translations for a single table across all languages"""
    english_qa_path, table_id, context_table, english_qa_list = table_info
    
    results = {
        "table_id": table_id,
        "completed": 0,
        "skipped": 0, # Skipped during this run (e.g., already in progress or completed on disk)
        "failed": 0
    }
    
    for lang_code, lang_name in cfg.LANGUAGES.items():
        output_dir = cfg.OUTPUT_DIR / lang_code
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / english_qa_path.name
        
        # Check if already completed - both in shared dict and on disk
        completion_key = f"{lang_code}_{table_id}"
        
        # Check shared dictionary first (in-progress tracking)
        with lock:
            if completion_key in completed_dict and completed_dict[completion_key] == True:
                results["skipped"] += 1
                continue
        
        # Double-check on disk in case file exists but wasn't in initial load
        if output_path.exists():
            with lock:
                completed_dict[completion_key] = True
            results["skipped"] += 1
            continue
        
        # Mark as in-progress to prevent other processes from picking it up
        with lock:
            if completion_key in completed_dict:
                # Another process grabbed it while we were checking
                results["skipped"] += 1
                continue
            # Mark as in-progress
            completed_dict[completion_key] = "in_progress"
        
        print(f"\n[Worker #{worker_id}][{table_id}] Translating to {lang_name} ({lang_code})")
        
        translated_qa_list = []
        failed_qa_count = 0
        
        for idx, english_qa_pair in enumerate(english_qa_list):
            try:
                translator = QATranslator(
                    english_qa_pair, 
                    context_table, 
                    table_id=table_id,
                    worker_id=worker_id
                )
                
                if lang_code == "en":
                    new_qa_pair = english_qa_pair.copy()
                    translated_qa_list.append(new_qa_pair)
                else:
                    translation_result = translator.translate(lang_name)
                    
                    if translation_result:
                        new_qa_pair = english_qa_pair.copy()
                        new_qa_pair['question'] = translation_result.translated_question
                        new_qa_pair['answer'] = translation_result.translated_answer
                        translated_qa_list.append(new_qa_pair)
                    else:
                        failed_qa_count += 1
                        print(f"  [Worker #{worker_id}][{table_id}] Failed to translate QA pair {idx + 1}")
            
            except Exception as e:
                failed_qa_count += 1
                print(f"  [Worker #{worker_id}][{table_id}] Exception on QA pair {idx + 1}: {e}")
                
                if "all api keys have exceeded" in str(e).lower():
                    print(f"  [Worker #{worker_id}][{table_id}] Stopping translation due to quota exhaustion.")
                    break
        
        # Save translated QA pairs
        if translated_qa_list:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(translated_qa_list, f, indent=2, ensure_ascii=False)
            
            # Mark as complete (thread-safe)
            with lock:
                completed_dict[completion_key] = True
            
            success_rate = len(translated_qa_list) / len(english_qa_list) * 100
            print(f"  [Worker #{worker_id}][{table_id}] Saved {len(translated_qa_list)}/{len(english_qa_list)} "
                  f"({success_rate:.1f}%) QA pairs for {lang_name}")
            
            if failed_qa_count > 0:
                print(f"  [Worker #{worker_id}][{table_id}] {failed_qa_count} QA pairs failed translation")
            
            results["completed"] += 1
        else:
            print(f"  [Worker #{worker_id}][{table_id}] No QA pairs successfully translated for {lang_name}")
            # Remove in-progress marker on failure
            with lock:
                if completion_key in completed_dict and completed_dict[completion_key] == "in_progress":
                    del completed_dict[completion_key]
            results["failed"] += 1
    
    return results


def load_table_data(english_qa_path, table_type):
    """Load table data and QA pairs, filtering by table_type (e.g., 'wiki' or 'finqa')"""
    table_id = english_qa_path.stem.replace("_qa", "")
    
    # Filter only tables of the specified type
    if not table_id.startswith(table_type):
        return None
    
    # Load context table
    table_path = cfg.TABLES_DIR / f"{table_id}.json"
    if not table_path.exists():
        print(f"[WARN] Context table not found for {table_id}")
        return None
    
    with open(table_path, 'r', encoding='utf-8') as f:
        context_table = json.load(f)
    
    # Load English QA pairs
    with open(english_qa_path, 'r', encoding='utf-8') as f:
        english_qa_list = json.load(f)
    
    return (english_qa_path, table_id, context_table, english_qa_list)


def load_completed_translations(table_type):
    """Load already completed translations from disk with detailed statistics, filtering by table_type"""
    completed = {}
    per_language_stats = defaultdict(set)
    
    for lang_code in cfg.LANGUAGES.keys():
        lang_dir = cfg.OUTPUT_DIR / lang_code
        if lang_dir.exists():
            for json_file in lang_dir.glob("*.json"):
                table_id = json_file.stem.replace("_qa", "")
                # Only track tables of the specified type
                if table_id.startswith(table_type):
                    completion_key = f"{lang_code}_{table_id}"
                    completed[completion_key] = True
                    per_language_stats[lang_code].add(table_id)
    
    return completed, per_language_stats


def print_initial_statistics(per_language_stats, total_target_tables, logger, table_type):
    """Print detailed statistics of already completed translations for the specified table type"""
    
    header = "="*80
    logger.info(header)
    logger.info(f"  INITIAL TRANSLATION STATUS FOR {table_type.upper()} TABLES")
    logger.info(header)
    
    print("\n" + header)
    print(f"  INITIAL TRANSLATION STATUS FOR {table_type.upper()} TABLES")
    print(header)
    
    # Overall summary
    total_possible = total_target_tables * len(cfg.LANGUAGES)
    total_completed = sum(len(tables) for tables in per_language_stats.values())
    
    summary = f"Overall Progress: {total_completed}/{total_possible} translations completed ({total_completed/total_possible*100:.1f}%)"
    logger.info(summary)
    print(summary)
    
    logger.info(f"Total {table_type} tables found: {total_target_tables}")
    print(f"Total {table_type} tables found: {total_target_tables}")
    
    logger.info(f"Target languages: {len(cfg.LANGUAGES)}")
    print(f"Target languages: {len(cfg.LANGUAGES)}\n")
    
    # Per-language breakdown
    logger.info("\nPer-Language Completion Status:")
    print("Per-Language Completion Status:")
    print("-" * 80)
    
    # Sort languages by completion count (descending)
    sorted_langs = sorted(
        per_language_stats.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for lang_code, completed_tables in sorted_langs:
        lang_name = cfg.LANGUAGES[lang_code]
        completed_count = len(completed_tables)
        percentage = (completed_count / total_target_tables * 100) if total_target_tables > 0 else 0
        
        status_line = f"  {lang_name:20s} ({lang_code}): {completed_count:4d}/{total_target_tables} tables ({percentage:5.1f}%)"
        logger.info(status_line)
        print(status_line)
    
    # Show languages with missing translations
    print("\n" + "-" * 80)
    logger.info("\nLanguages Needing Translations:")
    print("Languages Needing Translations:")
    
    incomplete_langs = [(lang_code, cfg.LANGUAGES[lang_code], len(per_language_stats.get(lang_code, set())))
                        for lang_code in cfg.LANGUAGES.keys()
                        if len(per_language_stats.get(lang_code, set())) < total_target_tables]
    
    if incomplete_langs:
        for lang_code, lang_name, completed_count in incomplete_langs:
            remaining = total_target_tables - completed_count
            status_line = f"  {lang_name:20s} ({lang_code}): {remaining} tables remaining"
            logger.info(status_line)
            print(status_line)
    else:
        complete_msg = "  All languages are fully translated!"
        logger.info(complete_msg)
        print(complete_msg)
    
    print(header + "\n")
    logger.info(header + "\n")


def main():
    start_time = time.time()

    table_type = "wiki" 
    log_dir = Path("logs/translation")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log = log_dir / f"main_process_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [MAIN] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("  STARTING PHASE 2.2: QA PAIR TRANSLATION")
    logger.info("       (MULTIPROCESSING MODE)")
    logger.info("="*60)
    logger.info(f"Main log file: {main_log}")
    logger.info(f"Rate limit: {getattr(cfg, 'REQUESTS_PER_MINUTE', 10)} requests/minute per key")
    logger.info(f"Wait time after batch: 30 seconds")
    logger.info(f"Total API keys: {len(cfg.GEMINI_API_KEYS)}")
    logger.info(f"Target languages: {len(cfg.LANGUAGES)}")
    logger.info(f"Parallel workers: {min(5, len(cfg.GEMINI_API_KEYS))}")
    logger.info(f"Processing table type: {table_type.upper()}") # Added
    logger.info("="*60)
    
    print("="*60)
    print("  STARTING PHASE 2.2: QA PAIR TRANSLATION")
    print("       (MULTIPROCESSING MODE)")
    print("="*60)
    print(f"Main log file: {main_log}")
    print(f"Worker logs will be in: {log_dir}/")
    print(f"Rate limit: {getattr(cfg, 'REQUESTS_PER_MINUTE', 10)} requests/minute per key")
    print(f"Wait time after batch: 30 seconds")
    print(f"Total API keys: {len(cfg.GEMINI_API_KEYS)}")
    print(f"Target languages: {len(cfg.LANGUAGES)}")
    print(f"Parallel workers: {min(5, len(cfg.GEMINI_API_KEYS))}")
    print(f"Processing table type: {table_type.upper()}") # Added
    print("="*60 + "\n")
    
    # Find all English QA files
    english_qa_paths = sorted(list(cfg.INPUT_QA_DIR.glob("*.json")))
    logger.info(f"Found {len(english_qa_paths)} total QA files")
    print(f"Found {len(english_qa_paths)} total QA files")
    
    # Load table data (filter by table_type)
    logger.info(f"Loading {table_type} tables...")
    print(f"Loading {table_type} tables...")
    table_data_list = []
    for path in english_qa_paths:
        data = load_table_data(path, table_type) # Passed table_type
        if data:
            table_data_list.append(data)
    
    logger.info(f"Found {len(table_data_list)} {table_type} tables to process")
    print(f"Found {len(table_data_list)} {table_type} tables to process\n")
    
    if not table_data_list:
        logger.warning(f"No {table_type} tables found to process!")
        print(f"No {table_type} tables found to process!")
        return
    
    # Load completed translations with detailed statistics
    logger.info("Scanning for completed translations...")
    print("Scanning for completed translations...")
    initial_completed, per_language_stats = load_completed_translations(table_type) # Passed table_type
    
    # Print detailed initial statistics
    print_initial_statistics(per_language_stats, len(table_data_list), logger, table_type) # Passed table_type
    
    # Filter out table-language pairs that are already completed
    tables_needing_work = []
    
    for table_data in table_data_list:
        _, table_id, _, _ = table_data
        
        # Check if this table needs any language translations
        needs_work = False
        for lang_code in cfg.LANGUAGES.keys():
            completion_key = f"{lang_code}_{table_id}"
            if completion_key not in initial_completed:
                needs_work = True
                break
        
        if needs_work:
            tables_needing_work.append(table_data)
    
    fully_completed_count = len(table_data_list) - len(tables_needing_work)
    
    logger.info(f"Tables Status:")
    logger.info(f"  - {fully_completed_count} {table_type} tables fully completed (all {len(cfg.LANGUAGES)} languages)")
    logger.info(f"  - {len(tables_needing_work)} {table_type} tables need work (missing some languages)")
    
    print(f"Tables Status:")
    print(f"  - {fully_completed_count} {table_type} tables fully completed (all {len(cfg.LANGUAGES)} languages)")
    print(f"  - {len(tables_needing_work)} {table_type} tables need work (missing some languages)\n")
    
    if not tables_needing_work:
        logger.info(f"All {table_type} tables are already fully translated for all languages!")
        print(f"All {table_type} tables are already fully translated for all languages!")
        return
    
    # Create shared state for multiprocessing
    manager = Manager()
    completed_dict = manager.dict(initial_completed)
    lock = manager.Lock()
    
    # Calculate remaining tasks (tasks that were not initially completed)
    remaining_tasks_to_attempt = 0
    for table_data in tables_needing_work:
        _, table_id, _, _ = table_data
        for lang_code in cfg.LANGUAGES.keys():
            completion_key = f"{lang_code}_{table_id}"
            if completion_key not in initial_completed:
                remaining_tasks_to_attempt += 1
    
    logger.info(f"Remaining translation tasks to attempt: {remaining_tasks_to_attempt}")
    print(f"Remaining translation tasks to attempt: {remaining_tasks_to_attempt}\n")
    
    # Process tables in parallel
    num_workers = min(5, len(cfg.GEMINI_API_KEYS), len(tables_needing_work))
    logger.info(f"Starting {num_workers} parallel workers for {table_type} tables...")
    logger.info("Distribution strategy: Round-robin assignment to ensure each worker gets unique tables")
    print(f"Starting {num_workers} parallel workers for {table_type} tables...")
    print(f"Distribution strategy: Round-robin assignment to ensure each worker gets unique tables\n")
    
    # Assign tables to workers in round-robin fashion
    worker_assignments = {i+1: [] for i in range(num_workers)}
    for idx, table in enumerate(tables_needing_work):
        worker_id = (idx % num_workers) + 1
        worker_assignments[worker_id].append(table)
    
    # Show distribution
    logger.info("Worker assignments:")
    print("Worker assignments:")
    for worker_id, tables in worker_assignments.items():
        table_ids = [t[1] for t in tables]
        log_msg = f"  Worker #{worker_id} (API Key #{worker_id}): {len(tables)} {table_type} tables"
        logger.info(log_msg)
        print(log_msg)
        if len(tables) <= 5:
            tables_str = f"    Tables: {', '.join(table_ids)}"
            logger.info(tables_str)
            print(tables_str)
        else:
            tables_str = f"    First 5 tables: {', '.join(table_ids[:5])}..."
            logger.info(tables_str)
            print(tables_str)
    print()
    
    # Flatten assignments with worker IDs
    tables_with_ids = []
    for worker_id, tables in worker_assignments.items():
        for table in tables:
            tables_with_ids.append((worker_id, table))
    
    # Initialize statistics for this run
    total_possible_tasks_for_type = len(cfg.LANGUAGES) * len(table_data_list)
    initial_completed_tasks = sum(len(tables) for tables in per_language_stats.values())

    stats = {
        "total_possible_tasks": total_possible_tasks_for_type,
        "initial_completed": initial_completed_tasks,
        "newly_completed": 0,
        "skipped_during_run": 0, # Tasks skipped because another worker picked it up or it was completed on disk during the run
        "failed": 0
    }
    
    logger.info(f"Starting parallel processing of {len(tables_with_ids)} {table_type} table assignments...")
    print(f"Starting parallel processing of {len(tables_with_ids)} {table_type} table assignments...\n")
    
    with Pool(processes=num_workers) as pool:
        process_func = partial(
            process_single_table_wrapper,
            completed_dict=completed_dict,
            lock=lock,
            total_languages=len(cfg.LANGUAGES)
        )
        
        results = list(tqdm(
            pool.imap(process_func, tables_with_ids),
            total=len(tables_with_ids),
            desc=f"Processing {table_type} tables"
        ))
    
    # Aggregate results from all workers
    for result in results:
        stats["newly_completed"] += result["completed"]
        stats["skipped_during_run"] += result["skipped"]
        stats["failed"] += result["failed"]
    
    total_completed = stats["initial_completed"] + stats["newly_completed"]
    total_skipped_overall = stats["initial_completed"] + stats["skipped_during_run"]
    total_failed = stats["failed"]
    
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    
    logger.info("="*60)
    logger.info(f"  PHASE 2.2: QA PAIR TRANSLATION COMPLETE FOR {table_type.upper()} TABLES")
    logger.info("="*60)
    logger.info(f"Total time: {elapsed_minutes:.2f} minutes")
    logger.info(f"Statistics for {table_type} tables:")
    logger.info(f"  Total possible tasks: {stats['total_possible_tasks']}")
    logger.info(f"  Total completed: {total_completed} ({total_completed / stats['total_possible_tasks'] * 100:.1f}%)")
    logger.info(f"    (Initial scan: {stats['initial_completed']}, Newly completed: {stats['newly_completed']})")
    logger.info(f"  Total skipped (during run): {stats['skipped_during_run']} ({stats['skipped_during_run'] / stats['total_possible_tasks'] * 100:.1f}%)")
    logger.info(f"  Total failed: {total_failed} ({total_failed / stats['total_possible_tasks'] * 100:.1f}%)")
    logger.info("="*60)
    
    print("\n" + "="*60)
    print(f"  PHASE 2.2: QA PAIR TRANSLATION COMPLETE FOR {table_type.upper()} TABLES")
    print("="*60)
    print(f"Total time: {elapsed_minutes:.2f} minutes")
    print(f"\nStatistics for {table_type} tables:")
    print(f"  Total possible tasks: {stats['total_possible_tasks']}")
    print(f"  Total completed: {total_completed} ({total_completed / stats['total_possible_tasks'] * 100:.1f}%)")
    print(f"    (Initial scan: {stats['initial_completed']}, Newly completed: {stats['newly_completed']})")
    print(f"  Total skipped (during run): {stats['skipped_during_run']} ({stats['skipped_during_run'] / stats['total_possible_tasks'] * 100:.1f}%)")
    print(f"  Total failed: {total_failed} ({total_failed / stats['total_possible_tasks'] * 100:.1f}%)")
    print("="*60)
    print(f"\nLog files location: {log_dir}/")
    print(f"Main log: {main_log}")
    print("="*60)


if __name__ == '__main__':
    main()