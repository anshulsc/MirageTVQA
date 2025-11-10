import json
import os
from pathlib import Path
from collections import defaultdict

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

def check_images_exist(base_path, table_id, lang_code):
    """
    Check if clean and noise images exist for a given table_id and language code.
    Returns: (has_clean, noise_count)
    """
    table_path = Path(base_path) / "images" / table_id
    
    # Check clean image
    clean_path = table_path / "clean" / f"{lang_code}_clean.jpg"
    has_clean = clean_path.exists()
    
    # Check noise images (1-3) in the noise folder
    noise_count = 0
    noise_path = table_path / "noise"
    for i in range(1, 4):
        noise_file = noise_path / f"{lang_code}_noise{i}.jpg"
        if noise_file.exists():
            noise_count += 1
    
    return has_clean, noise_count

def process_jsonl(input_file, output_file, base_path):
    """
    Process JSONL file and filter entries based on image existence.
    """
    stats = {
        'total_entries': 0,
        'filtered_entries': 0,
        'kept_entries': 0,
        'unique_clean_images': set(),
        'unique_noise_images': set(),
        'per_language': defaultdict(lambda: {
            'total': 0,
            'kept': 0,
            'filtered': 0,
            'clean_images': 0,
            'noise_images': 0,
            'unique_clean': set(),
            'unique_noise': set()
        })
    }
    
    kept_entries = []
    filtered_entries = []
    
    # Read and process JSONL
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            stats['total_entries'] += 1
            entry = json.loads(line.strip())
            
            table_id = entry.get('table_id')
            lang_code = entry.get('language')
            
            # Check if images exist
            has_clean, noise_count = check_images_exist(base_path, table_id, lang_code)
            
            # Update per-language stats
            stats['per_language'][lang_code]['total'] += 1
            
            # Keep entry if clean image exists
            if has_clean:
                kept_entries.append(entry)
                stats['kept_entries'] += 1
                stats['per_language'][lang_code]['kept'] += 1
                stats['per_language'][lang_code]['clean_images'] += 1
                stats['per_language'][lang_code]['noise_images'] += noise_count
                
                # Track unique images
                clean_img_path = f"{table_id}/{lang_code}_clean"
                stats['unique_clean_images'].add(clean_img_path)
                stats['per_language'][lang_code]['unique_clean'].add(clean_img_path)
                
                for i in range(1, noise_count + 1):
                    noise_img_path = f"{table_id}/{lang_code}_noise{i}"
                    stats['unique_noise_images'].add(noise_img_path)
                    stats['per_language'][lang_code]['unique_noise'].add(noise_img_path)
            else:
                filtered_entries.append(entry)
                stats['filtered_entries'] += 1
                stats['per_language'][lang_code]['filtered'] += 1
    
    # Write filtered JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in kept_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return stats, filtered_entries

def print_statistics(stats):
    """
    Print detailed statistics to terminal.
    """
    print("\n" + "="*80)
    print("IMAGE FILTERING STATISTICS")
    print("="*80)
    
    # Overall stats
    print("\n📊 OVERALL STATISTICS:")
    print(f"  Total questions:   {stats['total_entries']:,}")
    print(f"  Kept questions:    {stats['kept_entries']:,} ({stats['kept_entries']/stats['total_entries']*100:.2f}%)")
    print(f"  Filtered out:      {stats['filtered_entries']:,} ({stats['filtered_entries']/stats['total_entries']*100:.2f}%)")
    
    print("\n🖼️  UNIQUE IMAGE STATISTICS:")
    total_unique_clean = len(stats['unique_clean_images'])
    total_unique_noise = len(stats['unique_noise_images'])
    total_unique_images = total_unique_clean + total_unique_noise
    print(f"  Unique clean images:  {total_unique_clean:,}")
    print(f"  Unique noise images:  {total_unique_noise:,}")
    print(f"  Total unique images:  {total_unique_images:,}")
    
    # Per-language stats
    print("\n" + "="*80)
    print("PER-LANGUAGE STATISTICS:")
    print("="*80)
    
    # Sort languages by code
    sorted_langs = sorted(stats['per_language'].items())
    
    total_clean = 0
    total_noise = 0
    
    for lang_code, lang_stats in sorted_langs:
        lang_name = LANGUAGES.get(lang_code, lang_code)
        print(f"\n🌍 {lang_name} ({lang_code}):")
        print(f"  Questions:         {lang_stats['total']:,} total | {lang_stats['kept']:,} kept | {lang_stats['filtered']:,} filtered")
        
        if lang_stats['kept'] > 0:
            unique_clean = len(lang_stats['unique_clean'])
            unique_noise = len(lang_stats['unique_noise'])
            print(f"  Clean images:      {lang_stats['clean_images']:,} total | {unique_clean:,} unique")
            print(f"  Noise images:      {lang_stats['noise_images']:,} total | {unique_noise:,} unique")
            total_clean += lang_stats['clean_images']
            total_noise += lang_stats['noise_images']
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"  Total languages:         {len(stats['per_language'])}")
    print(f"  Total clean images:      {total_clean:,} (references)")
    print(f"  Total noise images:      {total_noise:,} (references)")
    print(f"  Unique clean images:     {total_unique_clean:,}")
    print(f"  Unique noise images:     {total_unique_noise:,}")
    print(f"  Total unique images:     {total_unique_images:,}")
    print("="*80 + "\n")

def main():
    # Configuration
    base_path = "/Users/anshulsingh/Projects/eurips/MMTQA/mirage_data"  # Adjust this path as needed
    input_file = os.path.join(base_path, "multilingual_qa_dataset.jsonl")
    output_file = os.path.join(base_path, "multilingual_qa_dataset_filtered.jsonl")
    
    print(f"Processing {input_file}...")
    print(f"Output will be written to {output_file}")
    
    # Process JSONL
    stats, filtered_entries = process_jsonl(input_file, output_file, base_path)
    
    # Print statistics
    print_statistics(stats)
    
    # Optional: Save filtered entries to a separate file for review
    filtered_output = os.path.join(base_path, "filtered_out_entries.jsonl")
    with open(filtered_output, 'w', encoding='utf-8') as f:
        for entry in filtered_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Filtered JSONL saved to: {output_file}")
    print(f"📝 Filtered out entries saved to: {filtered_output}")

if __name__ == "__main__":
    main()