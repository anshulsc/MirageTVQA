
from pathlib import Path
from typing import Optional, Dict

from src.table_ocr.extractor_factory import ExtractorFactory
from src.configs import text_extraction_config as cfg

class DeepSeekOCRProcessor:
    
    def __init__(self, image_path: Path, source: str = 'default', metadata: Optional[Dict] = None):
        self._extractor = ExtractorFactory.create_extractor(
            image_path=image_path,
            source=source,
            metadata=metadata
        )
        self.image_path = self._extractor.image_path
        self.source = self._extractor.source
        self.metadata = self._extractor.metadata
    
    def process_image(self, image, **kwargs):
        return self._extractor.process_image(image, **kwargs)
    
    def extract_and_save(self, output_table_path: Path, output_meta_path: Path) -> Dict:
        return self._extractor.extract_and_save(output_table_path, output_meta_path)
    
    def validate_image(self, image):
        return self._extractor.validate_image(image)

    @classmethod
    def _ensure_model_loaded(cls):
        pass


def create_extractor(
    image_path: Path,
    source: str = 'default',
    metadata: Optional[Dict] = None,
    method: Optional[str] = None
):

    return ExtractorFactory.create_extractor(image_path, source, metadata, method)


# Utility functions
def get_available_methods():
    return ExtractorFactory.get_available_methods()


def get_current_method():
    return ExtractorFactory.get_current_method()


def get_method_config(method: Optional[str] = None):
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