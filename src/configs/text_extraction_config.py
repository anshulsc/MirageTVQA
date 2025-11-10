from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

VISUAL_IMAGES_DIR = Path("/data/asca/cache/hub/datasets--anshulsc--TableLingua/snapshots/ceac5f8505e5bd1ae268b1636e48a0df18329df2/images")
VISUAL_METADATA_DIR = PROCESSED_DATA_DIR / "visual_metadata"


EXTRACTED_DIR = PROCESSED_DATA_DIR / "tables_ocr"
EXTRACTED_TABLES_DIR = EXTRACTED_DIR / "tables"
EXTRACTED_METADATA_DIR = EXTRACTED_DIR / "metadata"


EXTRACTED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_METADATA_DIR.mkdir(parents=True, exist_ok=True)

DEEPSEEK_CONFIG = {
    'model_name': "deepseek-ai/DeepSeek-OCR",
    'model_implementation': "flash_attention_2",
    'use_safetensors': True,
    'model_precision': "bfloat16",
    
    'size_configs': {
        "Tiny": {
            "base_size": 512,
            "image_size": 512,
            "crop_mode": False,
            "description": "Fastest, lowest quality - for quick previews"
        },
        "Small": {
            "base_size": 640,
            "image_size": 640,
            "crop_mode": False,
            "description": "Fast processing with acceptable quality"
        },
        "Base": {
            "base_size": 1024,
            "image_size": 1024,
            "crop_mode": False,
            "description": "Balanced speed and quality"
        },
        "Large": {
            "base_size": 1280,
            "image_size": 1280,
            "crop_mode": False,
            "description": "High quality, slower processing"
        },
        "Gundam (Recommended)": {
            "base_size": 1024,
            "image_size": 640,
            "crop_mode": True,
            "description": "Optimized for documents - best quality/speed ratio"
        }
    },
    'default_size': "Gundam (Recommended)",
    
    'task_configs': {
        "🔍 Free OCR": {
            "prompt_template": "<image>\nFree OCR.",
            "requires_ref": False,
            "description": "Extract raw text from the image"
        },
        "📄 Convert to Markdown": {
            "prompt_template": "<image>\n<|grounding|>Convert the document to markdown.",
            "requires_ref": False,
            "description": "Convert document to Markdown format"
        },
        "📈 Parse Figure": {
            "prompt_template": "<image>\nParse the figure.",
            "requires_ref": False,
            "description": "Extract structured data from charts"
        },
        "📍 Locate Object by Reference": {
            "prompt_template": "<image>\nLocate <|ref|>{ref_text}<|/ref|> in the image.",
            "requires_ref": True,
            "description": "Find specific object or text"
        }
    },
    'default_task': "📄 Convert to Markdown",
    

    'save_results': True,
    'test_compress': True,
    'eval_mode': True,
    

    'bbox_color': "red",
    'bbox_width': 3,
    'bbox_coordinate_space': 1000,
    
    # Performance Settings
    'gpu_memory_cleanup': True,
}


DOCLING_CONFIG = {
    'source_configs': {
        'arxiv': {
            'use_tableformer': False,  # Fast mode for academic papers
            'description': 'Academic papers - fast TableFormer mode'
        },
        'finqa': {
            'use_tableformer': True,  # Accurate mode for financial tables
            'description': 'Financial tables - accurate TableFormer mode'
        },
        'wiki': {
            'use_tableformer': False,  # Fast mode for Wikipedia
            'description': 'Wikipedia tables - fast mode'
        },
        'default': {
            'use_tableformer': False,  # Balanced approach
            'description': 'Generic tables - fast mode'
        }
    },
    
    # Processing settings
    'use_first_table_only': True,  # If False, combines all tables from image
    'min_confidence_threshold': 0.3,
    'min_table_rows': 2,
    'min_table_cols': 2,
}


TESSERACT_CONFIG = {
    'tesseract_path': None,  

    'source_configs': {
        'arxiv': {
            'psm_mode': 6, 
            'threshold_type': 'adaptive',
            'threshold_value': 150,
            'description': 'Academic papers - adaptive thresholding'
        },
        'finqa': {
            'psm_mode': 6,  # Uniform block
            'threshold_type': 'binary',
            'threshold_value': 180,
            'description': 'Financial tables - high contrast binary'
        },
        'wiki': {
            'psm_mode': 6,
            'threshold_type': 'adaptive',
            'threshold_value': 150,
            'description': 'Wikipedia tables - adaptive mode'
        },
        'default': {
            'psm_mode': 6,
            'threshold_type': 'binary',
            'threshold_value': 150,
            'description': 'Generic tables - balanced approach'
        }
    },
    
    # Image preprocessing
    'enable_denoising': True,
    'min_confidence_threshold': 0.3,
    'min_table_rows': 2,
    'min_table_cols': 2,
}

# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================

# Number of parallel workers for processing
# Set to 1 for GPU-intensive methods (DeepSeek) to avoid conflicts
# Can increase for CPU-based methods (Tesseract, Docling)
MAX_WORKERS = 1

# Batch size - number of images to process in each batch
# Helps with progress tracking and allows for checkpoint-style processing
# Adjust based on available memory and dataset size
BATCH_SIZE = 100  # Process 100 images at a time


# ============================================================================
# IMAGE VALIDATION
# ============================================================================

MAX_IMAGE_SIZE_PIXELS = 25_000_000  # ~5000x5000 pixels
MIN_IMAGE_SIZE_PIXELS = 100  # Minimum 10x10 pixels
PROCESSING_TIMEOUT_SECONDS = 120  # Max time per image

ERROR_MESSAGES = {
    "no_image": "⚠️ Please upload an image first.",
    "missing_ref": "⚠️ For the 'Locate' task, you must provide reference text!",
    "image_too_large": "⚠️ Image too large. Please use an image smaller than 5000x5000 pixels.",
    "image_too_small": "⚠️ Image too small. Minimum size is 10x10 pixels.",
    "invalid_image": "⚠️ Invalid image file. Please upload a valid image.",
    "processing_timeout": "⚠️ Processing timeout. Please try a smaller image or lower resolution.",
    "gpu_oom": "⚠️ GPU out of memory. Try a smaller resolution or image size.",
    "model_error": "⚠️ Model inference error: {error}",
    "method_not_found": "⚠️ OCR method '{method}' not found. Available: deepseek, docling, tesseract"
}

AVAILABLE_METHODS = ['deepseek', 'docling', 'tesseract']

OCR_METHOD = 'docling' 

if OCR_METHOD not in AVAILABLE_METHODS:
    raise ValueError(
        f"Invalid OCR_METHOD '{OCR_METHOD}'. "
        f"Available methods: {', '.join(AVAILABLE_METHODS)}"
    )