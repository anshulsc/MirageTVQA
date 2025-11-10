"""
Tesseract Extractor
Traditional OCR using Tesseract with preprocessing
"""
import re
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
from PIL import Image
import pytesseract

from src.configs import text_extraction_config as cfg
from src.table_ocr.extractors.base_extractor import BaseExtractor


class TesseractExtractor(BaseExtractor):
    
    def __init__(self, image_path, source='default', metadata=None):
        super().__init__(image_path, source, metadata)
        
        # Get Tesseract config
        self.config = cfg.TESSERACT_CONFIG
        
        # Get source-specific config
        self.source_config = self.config['source_configs'].get(
            self.source,
            self.config['source_configs']['default']
        )
        
        # Set Tesseract path if configured
        if self.config['tesseract_path']:
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_path']
        
        # Image cache
        self.image_cv = None
        self.tsv_data = None
    
    def load_image_cv(self):
        """Lazy load image as OpenCV array."""
        if self.image_cv is None:
            self.image_cv = cv2.imread(str(self.image_path))
            
            if self.image_cv is None:
                raise ValueError(f"Failed to load image: {self.image_path}")
                
        return self.image_cv
    
    def preprocess_image(self) -> np.ndarray:
        """Apply source-specific preprocessing."""
        img = self.load_image_cv()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding based on source
        threshold_type = self.source_config['threshold_type']
        
        if threshold_type == 'binary':
            threshold_value = self.source_config['threshold_value']
            _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            
        elif threshold_type == 'adaptive':
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 2
            )
        else:
            # Fallback to simple binary
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Optional denoising (slower but better quality)
        if self.config['enable_denoising']:
            thresh = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return thresh
    
    def extract_text(self) -> str:
        """Extract text using OCR with optimized settings."""
        preprocessed = self.preprocess_image()
        
        # Source-specific PSM mode
        psm_mode = self.source_config['psm_mode']
        custom_config = f'--oem 1 --psm {psm_mode}'
        
        # Get detailed TSV data for structure analysis
        try:
            self.tsv_data = pytesseract.image_to_data(
                preprocessed, 
                config=custom_config, 
                output_type=pytesseract.Output.DICT
            )
        except:
            self.tsv_data = None
        
        # Also get plain text as fallback
        text = pytesseract.image_to_string(preprocessed, config=custom_config)
        
        return text
    
    def calculate_confidence(self, text: str) -> float:
        """Estimate OCR confidence based on text characteristics."""
        if not text:
            return 0.0
        
        total_chars = len(text)
        alphanumeric = sum(c.isalnum() for c in text)
        whitespace = sum(c.isspace() for c in text)
        
        if total_chars == 0:
            return 0.0
        
        # Higher confidence if more alphanumeric, some whitespace is good
        confidence = (alphanumeric + whitespace * 0.5) / total_chars
        return round(min(confidence, 1.0), 3)
    
    def parse_table_to_json(self, text: str) -> Dict:
        """Parse extracted text into structured table format."""
        
        # Strategy 1: Use TSV spatial data for structure
        if hasattr(self, 'tsv_data') and self.tsv_data:
            result = self._parse_from_tsv_spatial()
            if result['valid']:
                return result
        
        # Strategy 2: Fallback to text parsing
        return self._parse_from_text(text)
    
    def _parse_from_tsv_spatial(self) -> Dict:
        """Parse table using spatial clustering of TSV word data."""
        tsv = self.tsv_data
        
        # Filter valid words with confidence
        words = []
        for i in range(len(tsv['text'])):
            if int(tsv['conf'][i]) < 30:  # Skip very low confidence
                continue
            
            text = tsv['text'][i].strip()
            if not text:
                continue
            
            words.append({
                'text': text,
                'left': tsv['left'][i],
                'top': tsv['top'][i],
                'width': tsv['width'][i],
                'height': tsv['height'][i],
            })
        
        if not words:
            return {"columns": [], "data": [], "valid": False, "num_rows": 0, "num_cols": 0}
        
        # Group words into rows by vertical position
        words.sort(key=lambda w: (w['top'], w['left']))
        
        rows = []
        current_row = [words[0]]
        row_threshold = 15  # pixels
        
        for word in words[1:]:
            if abs(word['top'] - current_row[0]['top']) <= row_threshold:
                current_row.append(word)
            else:
                rows.append(current_row)
                current_row = [word]
        
        if current_row:
            rows.append(current_row)
        
        # For each row, cluster words into columns by horizontal gaps
        table_rows = []
        for row_words in rows:
            row_words.sort(key=lambda w: w['left'])
            
            cells = []
            current_cell = [row_words[0]]
            
            for i in range(1, len(row_words)):
                prev_word = row_words[i-1]
                curr_word = row_words[i]
                
                # Calculate gap between words
                gap = curr_word['left'] - (prev_word['left'] + prev_word['width'])
                
                # Large gap = new column
                if gap > 40:  # pixels
                    cells.append(' '.join(w['text'] for w in current_cell))
                    current_cell = [curr_word]
                else:
                    current_cell.append(curr_word)
            
            cells.append(' '.join(w['text'] for w in current_cell))
            table_rows.append(cells)
        
        if not table_rows:
            return {"columns": [], "data": [], "valid": False, "num_rows": 0, "num_cols": 0}
        
        # Normalize column count
        num_cols = max(len(row) for row in table_rows)
        table_rows = [row + [''] * (num_cols - len(row)) for row in table_rows]
        
        is_valid = (
            len(table_rows) >= self.config['min_table_rows'] and 
            num_cols >= self.config['min_table_cols']
        )
        
        return {
            "columns": list(range(num_cols)),
            "data": table_rows,
            "valid": is_valid,
            "num_rows": len(table_rows),
            "num_cols": num_cols
        }
    
    def _parse_from_text(self, text: str) -> Dict:
        """Fallback text-based parsing."""
        if not text:
            return {"columns": [], "data": [], "valid": False, "num_rows": 0, "num_cols": 0}
        
        # Split into lines
        lines = text.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return {"columns": [], "data": [], "valid": False, "num_rows": 0, "num_cols": 0}
        
        # Split cells by multiple spaces, tabs, or pipes
        split_pattern = re.compile(r'\s{3,}|\t+|\|')
        
        rows = []
        for line in lines:
            cells = split_pattern.split(line)
            cells = [cell.strip() for cell in cells if cell.strip()]
            if cells:
                rows.append(cells)
        
        if not rows:
            return {"columns": [], "data": [], "valid": False, "num_rows": 0, "num_cols": 0}
        
        # Remove noise: single-cell rows with very short content
        rows = [row for row in rows if len(row) > 1 or len(row[0]) > 3]
        
        if not rows:
            return {"columns": [], "data": [], "valid": False, "num_rows": 0, "num_cols": 0}
        
        # Normalize row lengths
        num_cols = max(len(row) for row in rows)
        rows = [row + [''] * (num_cols - len(row)) for row in rows]
        
        # Validate table dimensions
        is_valid = (
            len(rows) >= self.config['min_table_rows'] and 
            num_cols >= self.config['min_table_cols']
        )
        
        return {
            "columns": list(range(num_cols)),
            "data": rows,
            "valid": is_valid,
            "num_rows": len(rows),
            "num_cols": num_cols
        }
    
    def process_image(
        self,
        image: Image.Image,
        **kwargs
    ) -> Tuple[str, Optional[Image.Image]]:
        """
        Main processing function - extracts text using Tesseract.
        
        Args:
            image: Input PIL Image (not used directly, we use CV2)
            **kwargs: Additional arguments (ignored for Tesseract)
            
        Returns:
            Tuple of (text_result, result_image)
        """
        try:
            print("🏃 Running Tesseract extraction...")
            print(f"   Source: {self.source}")
            print(f"   Config: {self.source_config['description']}")
            
            # Extract text
            text = self.extract_text()
            confidence = self.calculate_confidence(text)
            
            # Parse to table structure
            table_data = self.parse_table_to_json(text)
            
            # Format output
            text_result = self._format_output(text, table_data, confidence)
            
            print(f"✅ Tesseract extraction complete.")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Table valid: {table_data['valid']}")
            
            return text_result, None  # Tesseract doesn't produce result images
            
        except Exception as e:
            error_msg = f"Tesseract extraction error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, None
    
    def _format_output(self, text: str, table_data: Dict, confidence: float) -> str:
        """Format extraction results as text."""
        lines = []
        lines.append(f"# Tesseract OCR Results (Confidence: {confidence:.3f})")
        lines.append("")
        
        if table_data['valid']:
            lines.append(f"## Structured Table")
            lines.append(f"Dimensions: {table_data['num_rows']} rows × {table_data['num_cols']} columns")
            lines.append("")
            for row in table_data['data']:
                row_text = " | ".join(str(cell) for cell in row)
                lines.append(row_text)
        else:
            lines.append("## Raw Text (No valid table structure)")
            lines.append("")
            lines.append(text)
        
        return "\n".join(lines)