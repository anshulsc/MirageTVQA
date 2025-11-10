import os
import shutil
from pathlib import Path
from collections import defaultdict
import re

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
    
    # Sino-Tibetic
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

def extract_lang_code(filename):
    """Extract language code from filename."""
    # Try to match known language codes at the start of filename
    for lang_code in LANGUAGES.keys():
        if filename.startswith(lang_code + "_"):
            return lang_code
    return None

def is_clean_image(filename):
    """Check if image has 'clean' suffix."""
    return "_clean" in filename.lower()

def organize_images(source_dir, target_dir):
    """Organize images into clean and noise subdirectories."""
    
    # Statistics tracking
    stats = {
        'clean': defaultdict(int),
        'noise': defaultdict(int)
    }
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Find all table_id directories (all subdirectories)
    table_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(table_dirs)} table directories to process\n")
    
    for table_dir in sorted(table_dirs):
        table_id = table_dir.name
        print(f"Processing: {table_id}")
        
        # Create target directories
        clean_dir = target_path / table_id / "clean"
        noise_dir = target_path / table_id / "noise"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noise_dir.mkdir(parents=True, exist_ok=True)
        
        # Track noise counts per language for this table
        noise_counts = defaultdict(int)
        
        # Get all image files
        image_files = [f for f in table_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        for img_file in image_files:
            filename = img_file.name
            lang_code = extract_lang_code(filename)
            
            if not lang_code:
                print(f"  Warning: Could not extract language code from {filename}")
                continue
            
            if is_clean_image(filename):
                # Clean image
                new_name = f"{lang_code}_clean{img_file.suffix}"
                target_file = clean_dir / new_name
                shutil.copy2(img_file, target_file)
                stats['clean'][lang_code] += 1
                print(f"  ✓ Clean: {filename} -> {new_name}")
            else:
                # Noise image
                noise_counts[lang_code] += 1
                noise_num = noise_counts[lang_code]
                new_name = f"{lang_code}_noise{noise_num}{img_file.suffix}"
                target_file = noise_dir / new_name
                shutil.copy2(img_file, target_file)
                stats['noise'][lang_code] += 1
                print(f"  ✓ Noise: {filename} -> {new_name}")
        
        print()
    
    return stats

def print_statistics(stats):
    """Print detailed statistics."""
    print("\n" + "="*70)
    print("STATISTICS SUMMARY")
    print("="*70)
    
    # Get all unique languages
    all_langs = set(stats['clean'].keys()) | set(stats['noise'].keys())
    
    print(f"\n{'Language':<30} {'Clean':<10} {'Noise':<10} {'Total':<10}")
    print("-"*70)
    
    total_clean = 0
    total_noise = 0
    
    for lang_code in sorted(all_langs):
        lang_name = LANGUAGES.get(lang_code, lang_code)
        clean_count = stats['clean'][lang_code]
        noise_count = stats['noise'][lang_code]
        total = clean_count + noise_count
        
        print(f"{lang_name:<30} {clean_count:<10} {noise_count:<10} {total:<10}")
        
        total_clean += clean_count
        total_noise += noise_count
    
    print("-"*70)
    print(f"{'TOTAL':<30} {total_clean:<10} {total_noise:<10} {total_clean + total_noise:<10}")
    print("="*70)
    
    # Additional statistics
    print(f"\nTotal languages processed: {len(all_langs)}")
    print(f"Total images: {total_clean + total_noise}")
    print(f"Clean images: {total_clean} ({total_clean/(total_clean+total_noise)*100:.1f}%)")
    print(f"Noise images: {total_noise} ({total_noise/(total_clean+total_noise)*100:.1f}%)")

def main():
    # Configuration
    source_directory = "/Users/anshulsingh/Projects/eurips/MMTQA/data/processed/visual_images"  # Source directory with table_id folders
    target_directory = "/Users/anshulsingh/Projects/eurips/MMTQA/mirage_data/images/"  # Target directory for organized structure
    
    print("Image Organization Script")
    print("="*70)
    print(f"Source: {source_directory}")
    print(f"Target: {target_directory}")
    print("="*70 + "\n")
    
    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"Error: Source directory '{source_directory}' not found!")
        return
    
    # Organize images
    stats = organize_images(source_directory, target_directory)
    
    # Print statistics
    print_statistics(stats)
    
    print(f"\n✅ Organization complete! Images saved to '{target_directory}'")

if __name__ == "__main__":
    main()