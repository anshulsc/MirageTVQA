"""
Extractor Compatibility Wrapper
Maintains backward compatibility while using the new factory pattern
This file can replace your existing extractor.py
"""
from pathlib import Path
from typing import Optional, Dict

from text_extraction.extractor_factory import ExtractorFactory
from src.configs import text_extraction_config as cfg


# Backward compatibility class wrapper
class DeepSeekOCRProcessor:
    """
    Compatibility wrapper that maintains the old DeepSeekOCRProcessor interface
    but now uses ExtractorFactory under the hood.
    
    This allows existing code to continue working without changes while
    using the new multi-method architecture.
    """
    
    def __init__(self, image_path: Path, source: str = 'default', metadata: Optional[Dict] = None):
        """
        Initialize processor with automatic method selection from config.
        
        Args:
            image_path: Path to the image file
            source: Source type (arxiv, finqa, wiki, default)
            metadata: Optional metadata dict from rendering phase
        """
        # Create the actual extractor using factory
        self._extractor = ExtractorFactory.create_extractor(
            image_path=image_path,
            source=source,
            metadata=metadata
        )
        
        # Expose common attributes for backward compatibility
        self.image_path = self._extractor.image_path
        self.source = self._extractor.source
        self.metadata = self._extractor.metadata
    
    def process_image(self, image, **kwargs):
        """
        Process image using the configured extraction method.
        
        Args:
            image: PIL Image object
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (text_result, result_image)
        """
        return self._extractor.process_image(image, **kwargs)
    
    def extract_and_save(self, output_table_path: Path, output_meta_path: Path) -> Dict:
        """
        Extract text from image and save results.
        
        Args:
            output_table_path: Path to save extracted table JSON
            output_meta_path: Path to save extraction metadata
            
        Returns:
            Dict with status, image name, source, and optional error
        """
        return self._extractor.extract_and_save(output_table_path, output_meta_path)
    
    def validate_image(self, image):
        """
        Validate image dimensions and format.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self._extractor.validate_image(image)
    
    # Class-level compatibility for model loading check
    @classmethod
    def _ensure_model_loaded(cls):
        """
        Compatibility method for model loading.
        The actual implementation depends on the selected method.
        """
        # This is handled automatically by the extractors themselves
        pass


# Factory function for direct use
def create_extractor(
    image_path: Path,
    source: str = 'default',
    metadata: Optional[Dict] = None,
    method: Optional[str] = None
):
    """
    Create an extractor instance using the factory.
    
    Args:
        image_path: Path to the image file
        source: Source type (arxiv, finqa, wiki, default)
        metadata: Optional metadata dict from rendering phase
        method: OCR method override (if None, uses config default)
        
    Returns:
        Extractor instance
        
    Example:
        >>> extractor = create_extractor(Path("image.png"), source='finqa')
        >>> result = extractor.extract_and_save(table_path, meta_path)
    """
    return ExtractorFactory.create_extractor(image_path, source, metadata, method)


# Utility functions
def get_available_methods():
    """
    Get list of available OCR methods.
    
    Returns:
        List of method names
        
    Example:
        >>> methods = get_available_methods()
        >>> print(methods)
        ['deepseek', 'docling', 'tesseract']
    """
    return ExtractorFactory.get_available_methods()


def get_current_method():
    """
    Get the currently configured OCR method.
    
    Returns:
        Method name string
        
    Example:
        >>> method = get_current_method()
        >>> print(f"Using: {method}")
        Using: deepseek
    """
    return ExtractorFactory.get_current_method()


def get_method_config(method: Optional[str] = None):
    """
    Get configuration for a specific method.
    
    Args:
        method: Method name (if None, uses current method from config)
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = get_method_config('deepseek')
        >>> print(config['model_name'])
        deepseek-ai/DeepSeek-OCR
    """
    method = method or cfg.OCR_METHOD
    method = method.lower()
    
    if method == 'deepseek':
        return cfg.DEEPSEEK_CONFIG
    elif method == 'docling':
        return cfg.DOCLING_CONFIG
    elif method == 'tesseract':
        return cfg.TESSERACT_CONFIG
    else:
        raise ValueError(f"Unknown method: {method}")


# Deprecated global processor instance (for extreme backward compatibility)
_processor = None

def get_processor():
    """
    DEPRECATED: Get or create global processor instance.
    
    This is kept for backward compatibility but should not be used in new code.
    Instead, create instances directly using create_extractor() or DeepSeekOCRProcessor().
    
    Raises:
        NotImplementedError: Always raises, as this pattern is deprecated
    """
    raise NotImplementedError(
        "get_processor() is deprecated. Use create_extractor() or "
        "DeepSeekOCRProcessor(image_path, source, metadata) instead."
    )


# Export public API
__all__ = [
    'DeepSeekOCRProcessor',      # Main compatibility class
    'create_extractor',           # Factory function
    'get_available_methods',      # Utility
    'get_current_method',         # Utility
    'get_method_config',          # Utility
]