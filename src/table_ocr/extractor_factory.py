"""
OCR Extractor Factory
Provides unified interface for different OCR methods
"""
from pathlib import Path
from typing import Optional, Dict

from src.configs import text_extraction_config as cfg
from src.table_ocr.extractors.deepseek_extractor import DeepSeekExtractor
from src.table_ocr.extractors.docling_extractor import DoclingExtractor
from src.table_ocr.extractors.tesseract_extractor import TesseractExtractor


class ExtractorFactory:
    
    @staticmethod
    def create_extractor(
        image_path: Path,
        source: str = 'default',
        metadata: Optional[Dict] = None,
        method: Optional[str] = None
    ):
        method = method or cfg.OCR_METHOD
        method = method.lower()
        
        if method == 'deepseek':
            return DeepSeekExtractor(image_path, source, metadata)
        elif method == 'docling':
            return DoclingExtractor(image_path, source, metadata)
        elif method == 'tesseract':
            return TesseractExtractor(image_path, source, metadata)
        else:
            raise ValueError(
                cfg.ERROR_MESSAGES["method_not_found"].format(method=method)
            )
    
    @staticmethod
    def get_available_methods():
        return cfg.AVAILABLE_METHODS
    
    @staticmethod
    def get_current_method():
        return cfg.OCR_METHOD