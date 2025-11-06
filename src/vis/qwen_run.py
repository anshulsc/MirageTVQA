import torch
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from datasets import load_dataset
from src.vis.qwen_config import *  # Import your Qwen config
from src.vis.qwen_extractor import Qwen25VLAttentionOutputExtractor
from src.vis.probe_trainer import ProbeTrainer  # Your existing probe trainer
from src.vis.analyze_results import ResultAnalyzer  # Your existing analyzer

def main():
    print("--- Loading Qwen2.5-VL Model and Processor ---")
    
    # Load Qwen2.5-VL model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    
    print("\n--- Loading Dataset ---")
    dataset = load_dataset(
        DATASET_NAME, 
        split=DATASET_SPLIT,
        streaming=False,
        trust_remote_code=True
    )
    dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(NUM_SAMPLES))
    
    print("\n--- Running Step 1: Feature Extraction ---")
    # Create a config object with all the constants
    class Config:
        pass
    
    config = Config()
    config.DEVICE = DEVICE
    config.NUM_LAYERS = NUM_LAYERS
    config.NUM_HEADS = NUM_HEADS
    config.HEAD_DIM = HEAD_DIM
    config.FEATURES_DIR = FEATURES_DIR
    config.LANGUAGE_PROMPTS = LANGUAGE_PROMPTS
    config.USE_MASKED_ATTENTION_FOR_PROBES = USE_MASKED_ATTENTION_FOR_PROBES
    config.VERBOSE = VERBOSE
    config.LOG_FILE = LOG_FILE
    
    extractor = Qwen25VLAttentionOutputExtractor(model, processor, config)
    extractor.process_dataset(dataset, LANG_A, LANG_B)
    
    print("\n--- Running Step 2: Training Probes ---")
    config.PROBE_DIR = PROBE_DIR
    config.RESULTS_DIR = RESULTS_DIR
    config.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT
    config.RANDOM_SEED = RANDOM_SEED
    
    trainer = ProbeTrainer(config)
    trainer.train_all_probes()

    print("\n--- Running Step 3: Analyzing Results ---")
    config.VISUALIZATION_DIR = VISUALIZATION_DIR
    config.TOP_K = TOP_K
    config.LANG_A = LANG_A
    config.LANG_B = LANG_B
    
    analyzer = ResultAnalyzer(config)
    analyzer.analyze()
    
    print("\n--- Pipeline Complete ---")
    print(f"Check the '{OUTPUT_DIR}' directory for all outputs.")

if __name__ == "__main__":
    main()