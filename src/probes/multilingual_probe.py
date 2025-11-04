import os
import json
import random
from collections import defaultdict
from tqdm import tqdm

class MultilingualProbeGenerator:

    def __init__(self, qa_dataset_path):
        self.qa_dataset_path = qa_dataset_path
        self.qa_data = self._load_jsonl(qa_dataset_path)
        print(f"Loaded {len(self.qa_data)} QA pairs from '{qa_dataset_path}'.")
        self.processed_qa = None

    def _load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def preprocess_qa_dataset(self):
        print("Preprocessing the main QA dataset to group translations by question_id...")
        grouped_data = defaultdict(dict)

        for qa_pair in tqdm(self.qa_data, desc="Grouping by question_id"):
            if qa_pair.get('question_type') == 'value':
                q_id = qa_pair.get('question_id')
                lang = qa_pair.get('language')
                if q_id and lang:
                    grouped_data[q_id][lang] = qa_pair
        
        self.processed_qa = grouped_data
        
        num_english_pivots = sum(1 for q_group in self.processed_qa.values() if 'en' in q_group)
        print(f"Preprocessing complete. Found {len(self.processed_qa)} unique question IDs.")
        print(f"Found {num_english_pivots} question groups with an English version.")


    def generate_bilingual_probe(self, lang1, lang2):
        if self.processed_qa is None:
            raise RuntimeError("QA dataset has not been preprocessed. Call `preprocess_qa_dataset()` first.")

        samples = []
        for q_id, lang_map in self.processed_qa.items():
            if lang1 in lang_map and lang2 in lang_map:
                qa_pair1 = lang_map[lang1]
                qa_pair2 = lang_map[lang2]

                samples.append({
                    "question_id": qa_pair1['question_id'],
                    "table_id": qa_pair1['table_id'],
                    "question": qa_pair1['question'],
                    "probe_label": lang1
                })
                samples.append({
                    "question_id": qa_pair2['question_id'],
                    "table_id": qa_pair2['table_id'],
                    "question": qa_pair2['question'],
                    "probe_label": lang2
                })
        return samples

    def generate(self, language_pairs, output_dir):
        self.preprocess_qa_dataset()
        
        print("\n--- Starting Multilingual Probe Dataset Generation ---")
        for lang1, lang2 in language_pairs:
            print(f"\nGenerating probe dataset for {lang1.upper()}-{lang2.upper()}...")
            bilingual_samples = self.generate_bilingual_probe(lang1, lang2)
            
            if not bilingual_samples:
                print(f"Warning: No parallel data found for {lang1}-{lang2}. Skipping.")
                continue

            random.shuffle(bilingual_samples)
            
            output_file = os.path.join(output_dir, f"probe_multilingual_{lang1}_{lang2}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(bilingual_samples, f, indent=2, ensure_ascii=False)
            
            print(f"Successfully generated {len(bilingual_samples)} samples ({len(bilingual_samples)//2} pairs).")
            print(f"Dataset saved to '{output_file}'.")
if __name__ == "__main__":
    TABLES_DIRECTORY = "./data/processed/tables"
    QA_DATASET_PATH = "hf_dataset/data/dataset_final.jsonl" 
    OUTPUT_DIRECTORY = "./data/processed/probes/"

    BILINGUAL_PAIRS_TO_GENERATE = [
       
        ('en', 'zh_cn'),
        ('en', 'ar'),
        ('en', 'hi'),
        ('en', 'ja_formal'),
        ('en', 'ko_formal'),
        ('en', 'es'),
        ('en', 'mr'),
        ('en', 'ru_formal'),
        ('en', 'id_formal'),
        ('en', 'it'), 
        ('en', 'sc'),
    ]
    
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    generator = MultilingualProbeGenerator(
        qa_dataset_path=QA_DATASET_PATH
    )
    generator.generate(
        language_pairs=BILINGUAL_PAIRS_TO_GENERATE,
        output_dir=OUTPUT_DIRECTORY
    )