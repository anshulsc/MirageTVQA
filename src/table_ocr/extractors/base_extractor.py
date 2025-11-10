"""
Base Extractor Class
Abstract base class defining the interface for all OCR extractors
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

from PIL import Image

from src.configs import text_extraction_config as cfg


class BaseExtractor(ABC):
    """Abstract base class for all OCR extractors."""
    
    def __init__(self, image_path: Path, source: str = 'default', metadata: Optional[Dict] = None):
        """
        Initialize the extractor.
        
        Args:
            image_path: Path to the image file
            source: Source type (arxiv, finqa, wiki, default)
            metadata: Optional metadata dict from rendering phase
        """
        self.image_path = Path(image_path)
        self.source = source
        self.metadata = metadata or {}
    
    @abstractmethod
    def process_image(
        self,
        image: Image.Image,
        **kwargs
    ) -> Tuple[str, Optional[Image.Image]]:
        """
        Main processing function - runs OCR inference.
        
        Args:
            image: Input PIL Image
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (text_result, result_image)
        """
        pass
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, Optional[str]]:
        """
        Validate image dimensions and format.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, cfg.ERROR_MESSAGES["no_image"]
        
        try:
            width, height = image.size
            total_pixels = width * height
            
            if total_pixels > cfg.MAX_IMAGE_SIZE_PIXELS:
                return False, cfg.ERROR_MESSAGES["image_too_large"]
            
            if total_pixels < cfg.MIN_IMAGE_SIZE_PIXELS:
                return False, cfg.ERROR_MESSAGES["image_too_small"]
            
            return True, None
            
        except Exception as e:
            return False, cfg.ERROR_MESSAGES["invalid_image"]
    
    def extract_and_save(
        self,
        output_table_path: Path,
        output_meta_path: Path
    ) -> Dict:
        """
        Extract text from image and save results to specified paths.
        
        Args:
            output_table_path: Path to save extracted table JSON
            output_meta_path: Path to save extraction metadata
            
        Returns:
            Dict with status, image name, source, and optional error
        """
        try:
            # Load image
            image = Image.open(self.image_path)
            
            # Validate image
            is_valid, error_msg = self.validate_image(image)
            if not is_valid:
                return {
                    'status': 'failed',
                    'image': self.image_path.name,
                    'source': self.source,
                    'error': error_msg
                }
            
            # Process image (method-specific)
            text_result, result_image = self.process_image(image)
            
            # Check for errors
            if any(err in text_result for err in ["ERROR", "failed", "invalid", "⚠️"]):
                return {
                    'status': 'failed',
                    'image': self.image_path.name,
                    'source': self.source,
                    'error': text_result
                }
            
            # Assess quality (basic heuristic)
            quality_score = len(text_result.strip())
            status = 'success' if quality_score > 50 else 'low_quality'
            
            # Create output directories
            output_table_path.parent.mkdir(parents=True, exist_ok=True)
            output_meta_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save table data
            table_data = {
                'text': text_result,
                'source': self.source,
                'image_path': str(self.image_path),
                'quality_score': quality_score,
                'extraction_method': self.__class__.__name__
            }
            
            with open(output_table_path, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            extraction_metadata = {
                'status': status,
                'source': self.source,
                'image_name': self.image_path.name,
                'text_length': len(text_result),
                'quality_score': quality_score,
                'extraction_method': self.__class__.__name__,
                'rendering_metadata': self.metadata
            }
            
            with open(output_meta_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)
            
            return {
                'status': status,
                'image': self.image_path.name,
                'source': self.source
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'image': self.image_path.name,
                'source': self.source,
                'error': str(e)
            }