import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Set, Tuple
from termcolor import cprint
import random

def load_completed_instances(resume_file_path: str) -> Set[Tuple[str, str]]:
    """
    Load already completed instances from a partial evaluation file.
    
    Args:
        resume_file_path: Path to the incomplete evaluation jsonl file.
        
    Returns:
        A set of tuples (question_id, image_filename) that have been completed.
    """
    resume_file = Path(resume_file_path)
    if not resume_file.exists():
        cprint(f"Resume file not found: {resume_file_path}. Starting fresh.", "yellow")
        return set()
    
    completed = set()
    try:
        with open(resume_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    qid = result.get("question_id")
                    img_fname = result.get("image_filename")
                    if qid and img_fname:
                        completed.add((qid, img_fname))
                except json.JSONDecodeError:
                    continue
        
        cprint(f"Found {len(completed)} completed instances in resume file.", "green")
        return completed
    except Exception as e:
        cprint(f"Error reading resume file: {e}. Starting fresh.", "yellow")
        return set()

def load_benchmark_data(
    data_file_path: str,
    images_root_dir: str,
    image_type: str,
    lang_code_filter: str,
    resume_from: str | None = None
) -> List[Dict]:
    """
    Loads benchmark data by pairing QA entries from a JSONL file with image files
    discovered on the disk based on table_id and language.

    Args:
        data_file_path: Path to the .jsonl file with QA pairs.
        images_root_dir: Path to the root 'images' directory.
        image_type: The subfolder to search within ('clean' or 'noise').
        lang_code_filter: The language code to filter by, or 'default'.
        resume_from: Optional path to incomplete evaluation file to resume from.

    Returns:
        A list of dictionaries for the evaluation set (excluding completed instances).
    """
    data_file = Path(data_file_path)
    images_root = Path(images_root_dir)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root directory not found: {images_root_dir}")

    # Load completed instances if resuming
    completed_instances = set()
    if resume_from:
        completed_instances = load_completed_instances(resume_from)
        cprint(f"Resuming from: {resume_from}", "cyan")

    evaluation_set = []
    skipped_count = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cprint(f"Processing {len(lines)} QA entries from {data_file.name}...", "cyan")
    for line in tqdm(lines, desc="Matching QA with images"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        # --- LANGUAGE FILTERING ---
        row_lang = row.get("language", "en")
        if lang_code_filter != "default" and row_lang != lang_code_filter:
            continue
        
        # --- DYNAMIC IMAGE DISCOVERY ---
        table_id = row.get("table_id")
        question_id = row.get("question_id")
        if not table_id or not question_id:
            continue

        # 1. Construct the target directory path for the images
        target_image_dir = images_root / table_id / image_type
        
        if not target_image_dir.is_dir():
            continue # Skip if the corresponding image folder doesn't exist

        # 2. Find all images in that directory matching the language code
        #    Pattern: en_clean.jpg, en_noise1.jpg, en_noise2.jpg, etc.
        image_search_pattern = f"{row_lang}_*.jpg"
        found_images = sorted(list(target_image_dir.glob(image_search_pattern)))

        # 3. For noise images, randomly select one; for clean, use all
        if image_type == "noise" and found_images:
            found_images = [random.choice(found_images)]
        
        # 4. Create an evaluation instance for each discovered image
        for image_path in found_images:
            image_filename = image_path.name
            
            # Skip if this instance was already completed
            if (question_id, image_filename) in completed_instances:
                skipped_count += 1
                continue
            
            instance = {
                "question_id": question_id,
                "table_id": table_id,
                "language": row_lang,
                "question": row.get("question"),
                "golden_answer": row.get("answer"),
                "reasoning_category": row.get("reasoning_category"),
                "question_type": row.get("question_type"),
                "image_filename": image_filename,
            }
            evaluation_set.append(instance)
            
    if lang_code_filter != "default":
        cprint(f"Filtered for language '{lang_code_filter}'.", "green")
    
    if resume_from and skipped_count > 0:
        cprint(f"Skipped {skipped_count} already completed instances.", "green")
        
    cprint(f"Created a total of {len(evaluation_set)} evaluation instances using '{image_type}' images.", "green")
    return evaluation_set