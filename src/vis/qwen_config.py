from pathlib import Path

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = "cuda" 
TORCH_DTYPE = "float16"  # Qwen2.5-VL works better with bfloat16

# Model Architecture (for Qwen2.5-VL-3B)
NUM_LAYERS = 36  # Qwen2.5-VL-3B has 36 layers
NUM_HEADS = 16   # Qwen2.5-VL-3B has 16 attention heads
HEAD_DIM = 128   # 2048 / 16 = 128

# Dataset Configuration
DATASET_NAME = "phiyodr/coco2017"
DATASET_SPLIT = "validation"
NUM_SAMPLES = 1000

# Language Configuration
LANG_A = "English"
LANG_B = "Spanish"

# Language prompts for different languages
LANGUAGE_PROMPTS = {
    "English": "What is in the image?",
    "Chinese": "图像中是什么?",
    "Spanish": "¿Qué hay en la imagen?",
    "Russian": "Что изображено на картинке?",
    "Portuguese": "O que há na imagem?",
    "Bulgarian": "Какво има на изображението?",
    "Hindi": "छवि में क्या है?",
    "German": "Was ist auf dem Bild?",
}

# Training Configuration
TRAIN_TEST_SPLIT = 0.8  
RANDOM_SEED = 42

# Analysis Configuration
TOP_K = 100  # Number of top language-specific heads to identify

# Output Directories
OUTPUT_DIR = Path("outputs_qwen25vl")
FEATURES_DIR = OUTPUT_DIR / "attention_outputs"
PROBE_DIR = OUTPUT_DIR / "probes"
RESULTS_DIR = OUTPUT_DIR / "results"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Create directories
for directory in [FEATURES_DIR, PROBE_DIR, RESULTS_DIR, VISUALIZATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Attention Masking Configuration
USE_MASKED_ATTENTION_FOR_PROBES = True
USE_STANDARD_ATTENTION_FOR_SHIFTS = True

# Logging Configuration
VERBOSE = True
LOG_FILE = OUTPUT_DIR / "experiment.log"