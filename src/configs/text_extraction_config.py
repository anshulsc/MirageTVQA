"""
Text Extraction Configuration
Centralized configuration for OCR methods, model settings, and processing parameters
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Input directories (where rendered images are stored)
VISUAL_IMAGES_DIR = PROCESSED_DATA_DIR / "visual_images"
VISUAL_METADATA_DIR = PROCESSED_DATA_DIR / "visual_metadata"

# Output directories (where extracted text/tables will be saved)
EXTRACTED_DIR = PROCESSED_DATA_DIR / "tables_ocr"
EXTRACTED_TABLES_DIR = EXTRACTED_DIR / "tables"
EXTRACTED_METADATA_DIR = EXTRACTED_DIR / "metadata"

# Create output directories
EXTRACTED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_METADATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEEPSEEK-OCR CONFIGURATION
# ============================================================================
DEEPSEEK_CONFIG = {
    'model_name': "deepseek-ai/DeepSeek-OCR",
    'model_implementation': "flash_attention_2", # if this not work then switch to "sdpa" or ""
    'use_safetensors': True,
    'model_precision': "bfloat16",
    
    # Resolution/Size Configurations
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
    
    # Task Type Configurations
    'task_configs': {
        "📝 Free OCR": {
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
        "🔍 Locate Object by Reference": {
            "prompt_template": "<image>\nLocate <|ref|>{ref_text}<|/ref|> in the image.",
            "requires_ref": True,
            "description": "Find specific object or text"
        }
    },
    'default_task': "📄 Convert to Markdown",
    
    # Processing Settings
    'save_results': True,
    'test_compress': True,
    'eval_mode': True,
    
    # Bounding Box Drawing Settings
    'bbox_color': "red",
    'bbox_width': 3,
    'bbox_coordinate_space': 1000,
    
    # Performance Settings
    'gpu_memory_cleanup': True,
}

# ============================================================================
# DOCLING CONFIGURATION
# ============================================================================
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

# ============================================================================
# TESSERACT CONFIGURATION
# ============================================================================
TESSERACT_CONFIG = {
    'tesseract_path': None,  # Set path if Tesseract is not in system PATH
    
    # Source-specific OCR configurations
    'source_configs': {
        'arxiv': {
            'psm_mode': 6,  # Uniform block of text
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
# COMMON SETTINGS (applicable to all methods)
# ============================================================================
MAX_WORKERS = 1  # Number of parallel workers (1 for GPU safety)

# Validation Settings
MAX_IMAGE_SIZE_PIXELS = 25_000_000  # ~5000x5000 pixels
MIN_IMAGE_SIZE_PIXELS = 100  # Minimum 10x10 pixels
PROCESSING_TIMEOUT_SECONDS = 120  # Max time per image

# Error Messages
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

# ============================================================================
# METHOD VALIDATION
# ============================================================================
AVAILABLE_METHODS = ['deepseek', 'docling', 'tesseract']

# ============================================================================
# OCR METHOD SELECTION
# ============================================================================
# Choose which OCR method to use: 'deepseek', 'docling', or 'tesseract'
OCR_METHOD = 'deepseek'  # Change this to switch methods

if OCR_METHOD not in AVAILABLE_METHODS:
    raise ValueError(
        f"Invalid OCR_METHOD '{OCR_METHOD}'. "
        f"Available methods: {', '.join(AVAILABLE_METHODS)}"
    )
