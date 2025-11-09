"""
OCR Extractor Factory
Provides unified interface for different OCR methods
"""
from pathlib import Path
from typing import Optional, Dict

from src.configs import text_extraction_config as cfg
from text_extraction.extractors.deepseek_extractor import DeepSeekExtractor
from text_extraction.extractors.docling_extractor import DoclingExtractor
from text_extraction.extractors.tesseract_extractor import TesseractExtractor


class ExtractorFactory:
    """Factory class to create appropriate extractor based on config."""
    
    @staticmethod
    def create_extractor(
        image_path: Path,
        source: str = 'default',
        metadata: Optional[Dict] = None,
        method: Optional[str] = None
    ):
        """
        Create and return the appropriate extractor instance.
        
        Args:
            image_path: Path to the image file
            source: Source type (arxiv, finqa, wiki, default)
            metadata: Optional metadata dict from rendering phase
            method: OCR method override (if None, uses config default)
            
        Returns:
            Extractor instance (DeepSeekExtractor, DoclingExtractor, or TesseractExtractor)
            
        Raises:
            ValueError: If method is invalid
        """
        # Use provided method or fall back to config
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
        """Return list of available OCR methods."""
        return cfg.AVAILABLE_METHODS
    
    @staticmethod
    def get_current_method():
        """Return currently configured OCR method."""
        return cfg.OCR_METHOD