from pathlib import Path


# MODEL_ID = "llava-hf/llava-1.5-7b-hf"
# DEVICE = "cuda" 
# TORCH_DTYPE = "float16"  


# NUM_LAYERS = 32
# NUM_HEADS = 32
# HEAD_DIM = 128  # 4096 / 32

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" 
TORCH_DTYPE = "float16"  # Qwen2.5-VL works better with bfloat16

# Model Architecture (for Qwen2.5-VL-3B)
# NUM_LAYERS = 36  # Qwen2.5-VL-3B has 36 layers
# NUM_HEADS = 16   # Qwen2.5-VL-3B has 16 attention heads
# HEAD_DIM = 128   # 2048 / 16 = 128


# # Model Architecture (for Qwen2.5-VL-7B)
NUM_LAYERS = 28  # 
NUM_HEADS = 28   # 
HEAD_DIM = 128   # 


DATASET_JSON_PATH = "/data/asca/MirageTVQA/data/processed/probes/probe_multilingual_en_mr.json"  
IMAGES_BASE_PATH = "/data/asca/cache/hub/datasets--anshulsc--TableLingua/snapshots/ceac5f8505e5bd1ae268b1636e48a0df18329df2/images/" 
NUM_SAMPLES = None 


PROMPT_TEMPLATE = "<image>\nUSER: {question}\nASSISTANT:"




TRAIN_TEST_SPLIT = 0.8  
RANDOM_SEED = 42


TOP_K = 100  

OUTPUT_DIR = Path(f"outputs/{MODEL_ID.split("/")[-1]}/multlingual_en_mr/")
FEATURES_DIR = OUTPUT_DIR / "attention_outputs"  
PROBE_DIR = OUTPUT_DIR / "probes"
RESULTS_DIR = OUTPUT_DIR / "results"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"


for directory in [FEATURES_DIR, PROBE_DIR, RESULTS_DIR, VISUALIZATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


USE_MASKED_ATTENTION_FOR_PROBES = True

USE_STANDARD_ATTENTION_FOR_SHIFTS = True


VERBOSE = True
LOG_FILE = OUTPUT_DIR / "experiment.log"