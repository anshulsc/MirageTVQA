import os
import json
import argparse
import re
from collections import Counter
from typing import Union, List, Dict, Tuple
import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# -------------------------
# Colors for console printing
# -------------------------
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

# -------------------------
# Language Codes
# -------------------------

# Supported language codes
LANGUAGES = {
    # Afro-Asiatic
    "ar": "Arabic (MSA)",
    # Austronesian
    "id_casual": "Indonesian (Casual)",
    "id_formal": "Indonesian (Formal)",
    "jv_krama": "Javanese (Krama - Polite)",
    "jv_ngoko": "Javanese (Ngoko - Casual)",
    "su_loma": "Sundanese",
    "tl": "Tagalog",
    # Indo-European
    "bn": "Bengali",
    "cs": "Czech",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "mr": "Marathi",
    "ru_formal": "Russian (Formal)",
    "sc": "Sardinian",
    "si_formal_spoken": "Sinhala",
    # Japonic
    "ja_formal": "Japanese (Formal)",
    # Koreanic
    "ko_formal": "Korean (Formal)",
    # Kra-Dai
    "th": "Thai",
    # Sino-Tibetan
    "nan": "Hokkien (Written)",
    "zh_cn": "Chinese (Mandarin)",
    # Turkic
    "az": "Azerbaijani"
}

# -------------------------
# Normalization Functions
# -------------------------

def normalize_item(item: Union[str, int, float, list]) -> Union[str, int, float, list]:
    if isinstance(item, list):
        return [normalize_item(e) for e in item]
    if isinstance(item, (int, float)):
        return int(item) if isinstance(item, float) and item.is_integer() else item
    if isinstance(item, str):
        # Remove commas and spaces
        clean_str = item.replace(",", "").strip()
        
        # Remove superscripts or Unicode characters in numbers
        clean_str = re.sub(r"[^\d\.\-eE]", "", clean_str)
        
        # Convert to number if possible
        try:
            if clean_str and (clean_str.replace('.', '', 1).replace('-', '', 1).replace('e', '', 1).replace('E', '', 1).isdigit() or re.match(r"^-?\d+(\.\d+)?([eE]-?\d+)?$", clean_str)):
                num = float(clean_str)
                return int(num) if num.is_integer() else round(num, 6)
        except (ValueError, OverflowError):
            pass
        
        return item.lower()
    return item

def normalize_element(element: Union[str, int, float, list, dict]) -> Union[str, int, float, list]:
    if isinstance(element, dict):
        values = [element[key] for key in element.keys()]
        return normalize_element(values)
    if isinstance(element, list):
        return [normalize_element(e) for e in element]
    if isinstance(element, str):
        element = re.sub(r'[{}]|[\w-]+:', '', element)
    return normalize_item(element)

def structure_to_string(data: Union[str, list, dict]) -> str:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            numbers = re.findall(r'\b\d+\b', data)
            words = re.findall(r'\b[a-zA-Z]+\b', data)
            data = numbers + words

    if isinstance(data, dict):
        data = data.get("data", data)

    normalized = normalize_element(data)

    def flatten(items):
        result = []
        for item in items if isinstance(items, list) else [items]:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(str(item))
        return result

    flattened = flatten(normalized)
    return " ".join(flattened)

# -------------------------
# Evaluation Metrics
# -------------------------

def compute_exact_match(prediction: str, truth: str) -> int:
    return int(prediction == truth)

def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    if not pred_tokens or not truth_tokens:
        return float(int(pred_tokens == truth_tokens))

    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    
    if overlap == 0:
        return 0.0
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

def compute_adaptive_bleu(prediction: str, references: List[str]) -> float:
    smoothing = SmoothingFunction().method1
    pred_tokens = prediction.split()
    L = len(pred_tokens)
    ref_tokens_list = [ref.split() for ref in references]

    if L == 0 or not ref_tokens_list:
        return 0.0

    bleu1 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    if L <= 3:
        return bleu1
    elif 4 <= L <= 7:
        return (bleu1 + bleu2) / 2
    else:
        return (bleu1 + bleu2 + bleu4) / 3

# -------------------------
# Language Extraction
# -------------------------

def extract_language_from_filename(filename: str) -> str:
    """Extract language code from image filename like 'ar_clean.jpg' or 'id_casual_noise1.jpg'"""
    if not filename:
        return "unknown"
    
    # Extract the base name without extension
    base_name = os.path.splitext(filename)[0]
    
    # Try to match each known language code
    # Sort by length (longest first) to match compound codes like 'id_casual' before 'id'
    sorted_lang_codes = sorted(LANGUAGES.keys(), key=len, reverse=True)
    
    for lang_code in sorted_lang_codes:
        # Pattern: language_code followed by _clean or _noise[1-3]
        pattern = f'^{re.escape(lang_code)}_(clean|noise[1-3]?)$'
        if re.match(pattern, base_name):
            return lang_code
    
    return "unknown"

# -------------------------
# Main Processing Loop
# -------------------------

def load_dataset(dataset_file: str) -> Dict[Tuple[str, str], dict]:
    """Load dataset JSONL and create a lookup dictionary by (question_id, language)"""
    dataset_lookup = {}
    
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                question_id = entry.get("question_id")
                language = entry.get("language", "unknown")
                if question_id:
                    # Use tuple of (question_id, language) as key
                    dataset_lookup[(question_id, language)] = entry
            except json.JSONDecodeError:
                continue
    
    return dataset_lookup

def process_evaluation(dataset_file: str, response_file: str) -> dict:
    """Process evaluation by matching dataset and responses"""
    
    # Load dataset
    print(f"{Colors.OKBLUE}Loading dataset from:{Colors.RESET} {Colors.BOLD}{dataset_file}{Colors.RESET}")
    dataset_lookup = load_dataset(dataset_file)
    print(f"{Colors.OKGREEN}Loaded {len(dataset_lookup)} questions from dataset{Colors.RESET}")
    
    results = []
    total = 0
    total_skipped = 0
    parse_errors = 0
    cuda_errors = 0
    table_errors = 0
    no_match = 0

    em_sum = 0
    f1_sum = 0
    bleu_sum = 0

    # For per-language metrics
    lang_metrics = {}

    print(f"{Colors.OKBLUE}Processing responses from:{Colors.RESET} {Colors.BOLD}{response_file}{Colors.RESET}")
    
    with open(response_file, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Processing responses"):
            line = line.strip()
            if not line:
                continue
            
            try:
                response_entry = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            question_id = response_entry.get("question_id")
            
            # STEP 1: Extract language from image filename in response
            image_filename = response_entry.get("image_filename", "")
            lang = extract_language_from_filename(image_filename)
            
            if lang == "unknown":
                print(f"{Colors.WARNING}Could not extract language from filename '{image_filename}' for question {question_id}{Colors.RESET}")
                no_match += 1
                continue
            
            # STEP 2: Look up in dataset using both question_id and language
            lookup_key = (question_id, lang)
            
            if lookup_key not in dataset_lookup:
                print(f"{Colors.WARNING}No match found for question_id='{question_id}' and language='{lang}'{Colors.RESET}")
                no_match += 1
                continue
            
            dataset_entry = dataset_lookup[lookup_key]
            
            # Check for errors in response
            response_raw = response_entry.get("model_response", "")
            if isinstance(response_raw, str) and any(err in response_raw for err in ["CUDA out of memory", "Table too large"]):
                total_skipped += 1
                if "CUDA out of memory" in response_raw:
                    cuda_errors += 1
                if "Table too large" in response_raw:
                    table_errors += 1
                continue

            total += 1

            # Get golden answer from dataset
            golden_answer = dataset_entry.get("answer", [])
            model_response = response_entry.get("model_response", [])
            
            gold_str = structure_to_string(golden_answer).lower()
            model_str = structure_to_string(model_response).lower()

            em = compute_exact_match(model_str, gold_str)
            f1 = compute_f1(model_str, gold_str)
            bleu = compute_adaptive_bleu(model_str, [gold_str])

            em_sum += em
            f1_sum += f1
            bleu_sum += bleu

            # Per-language aggregation
            if lang not in lang_metrics:
                lang_metrics[lang] = {"em_sum": 0, "f1_sum": 0, "bleu_sum": 0, "count": 0}
            lang_metrics[lang]["em_sum"] += em
            lang_metrics[lang]["f1_sum"] += f1
            lang_metrics[lang]["bleu_sum"] += bleu
            lang_metrics[lang]["count"] += 1

            # Store comprehensive result
            results.append({
                "question_id": question_id,
                "language": lang,
                "question": dataset_entry.get("question"),
                "question_type": dataset_entry.get("question_type"),
                "reasoning_category": dataset_entry.get("reasoning_category"),
                "golden_answer": golden_answer,
                "model_response": model_response,
                "model_name": response_entry.get("model_name"),
                "image_filename": image_filename,
                "exact_match": em,
                "f1_score": f1,
                "bleu_score": bleu,
                "gold_str_normalized": gold_str,
                "model_str_normalized": model_str
            })

    # Compute per-language averages
    per_language_metrics = {}
    for lang, stats in lang_metrics.items():
        count = stats["count"]
        per_language_metrics[lang] = {
            "exact_match": stats["em_sum"] / count if count else 0,
            "f1_score": stats["f1_sum"] / count if count else 0,
            "bleu_score": stats["bleu_sum"] / count if count else 0,
            "samples": count
        }

    overall_metrics = {
        "exact_match": em_sum / total if total > 0 else 0,
        "f1_score": f1_sum / total if total > 0 else 0,
        "bleu_score": bleu_sum / total if total > 0 else 0,
        "processed_samples": total,
        "skipped_samples": total_skipped,
        "no_match_samples": no_match,
        "parse_errors": parse_errors,
        "cuda_errors": cuda_errors,
        "table_errors": table_errors,
        "total_responses": total + total_skipped + parse_errors + no_match,
        "per_language_metrics": per_language_metrics
    }

    return {"overall_metrics": overall_metrics, "per_sample_results": results}

# -------------------------
# Main CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA responses against dataset with EM, F1, and Adaptive BLEU")
    parser.add_argument("--dataset-file", required=True, help="Path to dataset JSONL file")
    parser.add_argument("--response-file", required=True, help="Path to model responses JSONL file")
    parser.add_argument("--output-dir", default=".", help="Directory to save output files (default: current directory)")
    args = parser.parse_args()

    print(f"{Colors.HEADER}{'='*80}{Colors.RESET}")
    print(f"{Colors.HEADER}Starting Evaluation{Colors.RESET}")
    print(f"{Colors.HEADER}{'='*80}{Colors.RESET}\n")
    
    metrics = process_evaluation(args.dataset_file, args.response_file)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model name from first sample (all should have same model)
    model_name = "unknown_model"
    if metrics["per_sample_results"]:
        model_name = metrics["per_sample_results"][0].get("model_name", "unknown_model")
    
    # Generate output filenames
    response_basename = os.path.basename(args.response_file)
    response_name_without_ext = os.path.splitext(response_basename)[0]
    
    # File 1: Detailed results JSONL (metrics_<original_name>.jsonl)
    detailed_output_file = os.path.join(args.output_dir, f"metrics_{response_basename}")
    
    # File 2: Summary JSON (summary_<original_name_without_ext>.json)
    summary_output_file = os.path.join(args.output_dir, f"summary_{response_name_without_ext}.json")
    
    # Save detailed results as JSONL
    print(f"\n{Colors.OKBLUE}Saving detailed results...{Colors.RESET}")
    with open(detailed_output_file, "w", encoding="utf-8") as f:
        for result in metrics["per_sample_results"]:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Prepare summary with metadata
    summary = {
        "metadata": {
            "model_name": model_name,
            "dataset_file": os.path.basename(args.dataset_file),
            "response_file": response_basename,
            "total_samples": metrics["overall_metrics"]["processed_samples"]
        },
        "overall_metrics": {
            "exact_match": metrics["overall_metrics"]["exact_match"],
            "f1_score": metrics["overall_metrics"]["f1_score"],
            "bleu_score": metrics["overall_metrics"]["bleu_score"],
            "processed_samples": metrics["overall_metrics"]["processed_samples"],
            "skipped_samples": metrics["overall_metrics"]["skipped_samples"],
            "no_match_samples": metrics["overall_metrics"]["no_match_samples"],
            "parse_errors": metrics["overall_metrics"]["parse_errors"],
            "cuda_errors": metrics["overall_metrics"]["cuda_errors"],
            "table_errors": metrics["overall_metrics"]["table_errors"],
            "total_responses": metrics["overall_metrics"]["total_responses"]
        },
        "per_language_metrics": metrics["overall_metrics"]["per_language_metrics"]
    }
    
    # Save summary
    with open(summary_output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Print summary
    # -------------------------
    print(f"\n{Colors.HEADER}{'='*80}{Colors.RESET}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Overall Metrics:{Colors.RESET}")
    print(f"{Colors.HEADER}{'='*80}{Colors.RESET}")
    print(f"Model Name: {Colors.BOLD}{model_name}{Colors.RESET}")
    print(f"Processed Samples: {Colors.BOLD}{metrics['overall_metrics']['processed_samples']}{Colors.RESET}")
    print(f"Skipped Samples: {Colors.WARNING}{metrics['overall_metrics']['skipped_samples']}{Colors.RESET}")
    print(f"No Match Samples: {Colors.WARNING}{metrics['overall_metrics']['no_match_samples']}{Colors.RESET}")
    print(f"Parse Errors: {Colors.FAIL}{metrics['overall_metrics']['parse_errors']}{Colors.RESET}")
    print(f"\n{Colors.OKGREEN}Exact Match: {Colors.BOLD}{metrics['overall_metrics']['exact_match']*100:.2f}%{Colors.RESET}")
    print(f"{Colors.OKGREEN}Average F1: {Colors.BOLD}{metrics['overall_metrics']['f1_score']*100:.2f}%{Colors.RESET}")
    print(f"{Colors.OKGREEN}Average Adaptive BLEU: {Colors.BOLD}{metrics['overall_metrics']['bleu_score']*100:.2f}%{Colors.RESET}")

    print(f"\n{Colors.HEADER}{'='*80}{Colors.RESET}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}Per-Language Metrics:{Colors.RESET}")
    print(f"{Colors.HEADER}{'='*80}{Colors.RESET}")
    for lang, vals in sorted(metrics["overall_metrics"]["per_language_metrics"].items()):
        lang_name = LANGUAGES.get(lang, lang)
        print(f"{Colors.BOLD}{lang} ({lang_name}):{Colors.RESET} "
              f"EM={vals['exact_match']*100:.2f}%, "
              f"F1={vals['f1_score']*100:.2f}%, "
              f"BLEU={vals['bleu_score']*100:.2f}%, "
              f"Samples={vals['samples']}")
    
    print(f"\n{Colors.OKGREEN}{'='*80}{Colors.RESET}")
    print(f"{Colors.OKGREEN}Output files saved:{Colors.RESET}")
    print(f"  {Colors.OKCYAN}Detailed results: {detailed_output_file}{Colors.RESET}")
    print(f"  {Colors.OKCYAN}Summary: {summary_output_file}{Colors.RESET}")
    print(f"{Colors.OKGREEN}{'='*80}{Colors.RESET}\n")

if __name__ == "__main__":
    main()
"""
python evaluate_metrics.py \
    --dataset-file /home/anshulsc/links/scratch/TableLingua/dataset_combined_final.jsonl \
    --response-file /home/anshulsc/links/scratch/projects/MMTQA/data/processed/evaluation_results_new/google_gemma-3-4b-it_multitableqa_clean_default_20251022_042714.jsonl \
    --output-dir ./results

"""