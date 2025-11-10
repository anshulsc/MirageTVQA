#!/usr/bin/env python3
"""
Multilingual QA Dataset Processor
Processes translated QA pairs from multiple language folders and generates analysis
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
import sys

LANGUAGES = {
    # Afro-Asiatic
    "ar": "Arabic (MSA)",
    "he": "Hebrew",
    "am": "Amharic",
    
    # Austronesian
    "id_casual": "Indonesian (Casual)",
    "id_formal": "Indonesian (Formal)",
    "jv_krama": "Javanese (Krama - Polite)",
    "jv_ngoko": "Javanese (Ngoko - Casual)",
    "su_loma": "Sundanese",
    "tl": "Tagalog",
    "ms": "Malay",
    "fil": "Filipino",
    
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
    "fa": "Persian",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "pl": "Polish",
    "no": "Norwegian",
    "sv": "Swedish",
    "da": "Danish",
    "el": "Greek",
    "ur": "Urdu",
    "pb": "Punjabi",
    "np": "Nepali",
    "pt": "Portuguese",
    
    # Japonic
    "ja_formal": "Japanese (Formal)",
    
    # Koreanic
    "ko_formal": "Korean (Formal)",
    
    # Kra-Dai
    "th": "Thai",
    
    # Sino-Tibetan
    "nan": "Hokkien (Written)",
    "zh_cn": "Chinese (Mandarin)",
    "my": "Burmese",
    
    # Turkic
    "az": "Azerbaijani",
    "tr": "Turkish",
    
    # Austroasiatic
    "vi": "Vietnamese",
    
    # Dravidian
    "ta": "Tamil",
    "te": "Telugu",
}

LANGUAGE_FAMILIES = {
    "Afro-Asiatic": ["ar", "he", "am"],
    "Austronesian": ["id_casual", "id_formal", "jv_krama", "jv_ngoko", "su_loma", "tl", "ms", "fil"],
    "Indo-European": ["bn", "cs", "en", "es", "fr", "hi", "it", "mr", "ru_formal", "sc", 
                      "si_formal_spoken", "fa", "uk", "ro", "pl", "no", "sv", "da", "el", "ur", "pb", "np", "pt"],
    "Japonic": ["ja_formal"],
    "Koreanic": ["ko_formal"],
    "Kra-Dai": ["th"],
    "Sino-Tibetan": ["nan", "zh_cn", "my"],
    "Turkic": ["az", "tr"],
    "Austroasiatic": ["vi"],
    "Dravidian": ["ta", "te"],
}


def get_language_family(lang_code: str) -> str:
    """Get the language family for a given language code."""
    for family, langs in LANGUAGE_FAMILIES.items():
        if lang_code in langs:
            return family
    return "Unknown"


def process_json_file(file_path: Path, lang_code: str) -> List[Dict[str, Any]]:
    """Process a single JSON file and extract QA pairs."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = []
        for item in data:
            qa_pair = {
                "question_id": item.get("question_id"),
                "table_id": item.get("table_id"),
                "language": lang_code,
                "language_name": LANGUAGES.get(lang_code, lang_code),
                "language_family": get_language_family(lang_code),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "question_type": item.get("question_type"),
                "reasoning_category": item.get("reasoning_category"),
                "evidence_cells": item.get("evidence_cells"),
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return []


def process_all_languages(base_dir: str) -> tuple[List[Dict], Dict]:
    """Process all language folders and collect statistics."""
    base_path = Path(base_dir)
    all_qa_pairs = []
    stats = defaultdict(lambda: {
        'files': 0,
        'qa_pairs': 0,
        'question_types': Counter(),
        'reasoning_categories': Counter()
    })
    
    for lang_code in LANGUAGES.keys():
        lang_dir = base_path / lang_code
        
        if not lang_dir.exists():
            print(f"Warning: Directory not found for {lang_code}: {lang_dir}")
            continue
        
        json_files = list(lang_dir.glob("*.json"))
        
        if not json_files:
            print(f"Warning: No JSON files found in {lang_dir}")
            continue
        
        print(f"Processing {lang_code} ({LANGUAGES[lang_code]}): {len(json_files)} files")
        
        for json_file in json_files:
            qa_pairs = process_json_file(json_file, lang_code)
            all_qa_pairs.extend(qa_pairs)
            
            # Collect statistics
            stats[lang_code]['files'] += 1
            stats[lang_code]['qa_pairs'] += len(qa_pairs)
            
            for qa in qa_pairs:
                stats[lang_code]['question_types'][qa['question_type']] += 1
                stats[lang_code]['reasoning_categories'][qa['reasoning_category']] += 1
    
    return all_qa_pairs, dict(stats)


def save_jsonl(data: List[Dict], output_file: str):
    """Save data to JSONL format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n✓ Saved {len(data)} QA pairs to {output_file}")


def print_statistics(stats: Dict, all_qa_pairs: List[Dict]):
    """Print comprehensive statistics to terminal."""
    
    print("\n" + "="*80)
    print("MULTILINGUAL QA DATASET STATISTICS".center(80))
    print("="*80)
    
    # Overall statistics
    total_languages = len([lang for lang in stats if stats[lang]['qa_pairs'] > 0])
    total_qa_pairs = sum(stats[lang]['qa_pairs'] for lang in stats)
    total_files = sum(stats[lang]['files'] for lang in stats)
    
    print(f"\n{'OVERALL STATISTICS':^80}")
    print("-"*80)
    print(f"Total Languages Processed: {total_languages}/{len(LANGUAGES)}")
    print(f"Total JSON Files: {total_files}")
    print(f"Total QA Pairs: {total_qa_pairs:,}")
    print(f"Average QA Pairs per Language: {total_qa_pairs/total_languages:.1f}")
    
    # Language family statistics
    print(f"\n{'STATISTICS BY LANGUAGE FAMILY':^80}")
    print("-"*80)
    family_stats = defaultdict(lambda: {'languages': 0, 'qa_pairs': 0})
    
    for lang_code, lang_stats in stats.items():
        if lang_stats['qa_pairs'] > 0:
            family = get_language_family(lang_code)
            family_stats[family]['languages'] += 1
            family_stats[family]['qa_pairs'] += lang_stats['qa_pairs']
    
    for family in sorted(family_stats.keys()):
        fstats = family_stats[family]
        print(f"{family:20s}: {fstats['languages']:2d} languages, {fstats['qa_pairs']:5d} QA pairs")
    
    # Per-language statistics
    print(f"\n{'PER-LANGUAGE STATISTICS':^80}")
    print("-"*80)
    print(f"{'Language':<30} {'Code':<15} {'Files':>6} {'QA Pairs':>10}")
    print("-"*80)
    
    sorted_langs = sorted(stats.items(), key=lambda x: x[1]['qa_pairs'], reverse=True)
    
    for lang_code, lang_stats in sorted_langs:
        if lang_stats['qa_pairs'] > 0:
            print(f"{LANGUAGES[lang_code]:<30} {lang_code:<15} {lang_stats['files']:>6} {lang_stats['qa_pairs']:>10,}")
    
    # Question type distribution
    print(f"\n{'QUESTION TYPE DISTRIBUTION (OVERALL)':^80}")
    print("-"*80)
    
    question_type_counts = Counter()
    for qa in all_qa_pairs:
        question_type_counts[qa['question_type']] += 1
    
    for qtype, count in question_type_counts.most_common():
        percentage = (count / total_qa_pairs) * 100
        print(f"{qtype:<40} {count:>6,} ({percentage:>5.1f}%)")
    
    # Reasoning category distribution
    print(f"\n{'REASONING CATEGORY DISTRIBUTION (OVERALL)':^80}")
    print("-"*80)
    
    reasoning_counts = Counter()
    for qa in all_qa_pairs:
        if qa['reasoning_category']:
            reasoning_counts[qa['reasoning_category']] += 1
    
    for category, count in reasoning_counts.most_common():
        percentage = (count / total_qa_pairs) * 100
        print(f"{category:<40} {count:>6,} ({percentage:>5.1f}%)")
    
    # Missing languages
    missing_langs = [lang for lang in LANGUAGES if lang not in stats or stats[lang]['qa_pairs'] == 0]
    if missing_langs:
        print(f"\n{'MISSING OR EMPTY LANGUAGES':^80}")
        print("-"*80)
        for lang in missing_langs:
            print(f"  • {LANGUAGES[lang]} ({lang})")
    
    print("\n" + "="*80 + "\n")


def main():

    base_dir = "/Users/anshulsingh/Projects/eurips/MMTQA/data/processed/qa_pairs_translated/"  # Adjust this path as needed
    output_file = "/Users/anshulsingh/Projects/eurips/MMTQA/mirage_data/multilingual_qa_dataset.jsonl"
    
    print(f"Starting processing from directory: {base_dir}")
    
    # Process all languages
    all_qa_pairs, stats = process_all_languages(base_dir)
    
    if not all_qa_pairs:
        print("Error: No QA pairs found!")
        return
    
    # Save to JSONL
    save_jsonl(all_qa_pairs, output_file)
    
    # Print statistics
    print_statistics(stats, all_qa_pairs)
    
    print(f"✓ Processing complete!")


if __name__ == "__main__":
    main()