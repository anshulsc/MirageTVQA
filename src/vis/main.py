import torch
import json
from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from src.vis import config
from src.vis.feature_extractor import AttentionOutputExtractor, Qwen25VLAttentionOutputExtractor
from src.vis.probe_trainer import ProbeTrainer
from src.vis.analyze_results import ResultAnalyzer

def load_custom_dataset(json_path, num_samples=None):
    print(f"Loading dataset from {json_path}")
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    unique_labels = list(set(record["probe_label"] for record in dataset))
    print(f"Found {len(unique_labels)} unique probe labels: {unique_labels}")
    if num_samples is not None:
        import random
        random.seed(config.RANDOM_SEED)
        dataset = random.sample(dataset, min(num_samples, len(dataset)))
        print(f"Using {len(dataset)} samples")
    
    return dataset

def main(model_name="Llava"):
    
    if model_name == "Llava":
        print("--- Loading Model and Processor ---")
        model = LlavaForConditionalGeneration.from_pretrained(
            config.MODEL_ID, 
            torch_dtype=getattr(torch, config.TORCH_DTYPE), 
            low_cpu_mem_usage=True
        ).to(config.DEVICE)
        
        processor = AutoProcessor.from_pretrained(config.MODEL_ID)
        
    else:
        print("--- Loading Qwen Model and Processor ---")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
        processor = AutoProcessor.from_pretrained(
            config.MODEL_ID,
            trust_remote_code=True
        )
    
    print("\n--- Loading Dataset ---")
    dataset = load_custom_dataset(config.DATASET_JSON_PATH, config.NUM_SAMPLES)
    
    print("\n--- Running Step 1: Feature Extraction ---")
    if model_name == "Qwen":
        extractor = Qwen25VLAttentionOutputExtractor(model, processor, config)
    else:
        extractor = AttentionOutputExtractor(model, processor, config)
    
    extractor.process_dataset(dataset, config.IMAGES_BASE_PATH)
    
    print("\n--- Running Step 2: Training Probes ---")
    trainer = ProbeTrainer(config)
    trainer.train_all_probes()

    print("\n--- Running Step 3: Analyzing Results ---")
    analyzer = ResultAnalyzer(config)
    analyzer.analyze()
    
    print("\n--- Pipeline Complete ---")
    print(f"Check the '{config.OUTPUT_DIR}' directory for all outputs.")

if __name__ == "__main__":
    main(model_name="Qwen")