"""
Text Extraction - OCR Implementation Modules

This package contains different OCR method implementations:
- DeepSeekExtractor: High-quality OCR using DeepSeek-OCR model
- DoclingExtractor: Table extraction optimized with Docling
- TesseractExtractor: Traditional OCR with Tesseract

All extractors inherit from BaseExtractor and provide a consistent interface.
"""

from .base_extractor import BaseExtractor
from .deepseek_extractor import DeepSeekExtractor
from .docling_extractor import DoclingExtractor
from .tesseract_extractor import TesseractExtractor

__all__ = [
    'BaseExtractor',
    'DeepSeekExtractor',
    'DoclingExtractor',
    'TesseractExtractor'
]

__version__ = '2.0.0'