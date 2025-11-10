"""
Docling Extractor
Table extraction using Docling library
"""
from typing import Optional, Tuple, List, Dict
import json

from PIL import Image
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import PdfFormatOption

from src.configs import text_extraction_config as cfg
from src.table_ocr.extractors.base_extractor import BaseExtractor


class DoclingExtractor(BaseExtractor):
    """Docling implementation of the base extractor."""
    
    def __init__(self, image_path, source='default', metadata=None):
        super().__init__(image_path, source, metadata)
        
        # Get Docling config
        self.config = cfg.DOCLING_CONFIG
        
        # Get source-specific config
        self.source_config = self.config['source_configs'].get(
            self.source,
            self.config['source_configs']['default']
        )
        
        # Initialize Docling converter
        self._init_converter()
    
    def _init_converter(self):
        """Initialize Docling DocumentConverter with source-specific options."""
        # Configure pipeline options based on source
        pipeline_options = PdfPipelineOptions()
        
        # Use TableFormer for better table extraction
        if self.source_config['use_tableformer']:
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        else:
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        
        # Enable OCR for images
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options.lang = ['eng']
        
        # Configure based on source type
        if self.source == 'finqa':
            # Financial tables need high accuracy
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        elif self.source == 'arxiv':
            # Academic papers - balanced approach
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        
        # Create converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    
    def extract_tables(self) -> List[Dict]:
        """Extract tables from image using Docling."""
        try:
            # Convert image document
            result = self.converter.convert(str(self.image_path))
            
            # Extract tables from the document
            tables = []
            for table in result.document.tables:
                # Convert table to structured format
                table_data = self._parse_docling_table(table)
                if table_data:
                    tables.append(table_data)
            
            return tables
            
        except Exception as e:
            print(f"Error extracting tables with Docling: {e}")
            return []
    
    def _parse_docling_table(self, table) -> Optional[Dict]:
        """Parse Docling table object into structured format."""
        try:
            # Get table data as DataFrame
            table_data = table.export_to_dataframe()
            
            if table_data is None or table_data.empty:
                return None
            
            # Convert DataFrame to list of lists
            rows = table_data.values.tolist()
            
            # Get column headers
            columns = table_data.columns.tolist()
            
            # If columns are just indices, create generic column names
            if all(isinstance(col, int) for col in columns):
                columns = list(range(len(columns)))
            
            # Validate dimensions
            num_rows = len(rows)
            num_cols = len(columns)
            
            is_valid = (
                num_rows >= self.config['min_table_rows'] and 
                num_cols >= self.config['min_table_cols']
            )
            
            return {
                "columns": columns,
                "data": rows,
                "valid": is_valid,
                "num_rows": num_rows,
                "num_cols": num_cols
            }
            
        except Exception as e:
            print(f"Error parsing Docling table: {e}")
            return None
    
    def calculate_confidence(self, tables: List[Dict]) -> float:
        """Estimate extraction confidence based on table characteristics."""
        if not tables:
            return 0.0
        
        # Count valid tables
        valid_tables = sum(1 for t in tables if t.get('valid', False))
        
        if valid_tables == 0:
            return 0.0
        
        # Calculate confidence based on data completeness
        total_cells = 0
        filled_cells = 0
        
        for table in tables:
            if not table.get('valid', False):
                continue
                
            for row in table['data']:
                total_cells += len(row)
                filled_cells += sum(1 for cell in row if cell and str(cell).strip())
        
        if total_cells == 0:
            return 0.5  # Found structure but no data
        
        # Confidence is ratio of filled cells plus bonus for having valid tables
        data_completeness = filled_cells / total_cells
        table_bonus = min(valid_tables * 0.1, 0.3)  # Up to 30% bonus
        
        confidence = min(data_completeness + table_bonus, 1.0)
        return round(confidence, 3)
    
    def _combine_tables(self, tables: List[Dict]) -> Dict:
        """Combine multiple tables into one."""
        valid_tables = [t for t in tables if t.get('valid', False)]
        
        if not valid_tables:
            return {
                "columns": [], "data": [], "valid": False,
                "num_rows": 0, "num_cols": 0
            }
        
        if len(valid_tables) == 1:
            return valid_tables[0]
        
        # Combine tables vertically (stack rows)
        all_rows = []
        max_cols = max(t['num_cols'] for t in valid_tables)
        
        for table in valid_tables:
            for row in table['data']:
                # Pad row to match max columns
                padded_row = row + [''] * (max_cols - len(row))
                all_rows.append(padded_row)
        
        return {
            "columns": list(range(max_cols)),
            "data": all_rows,
            "valid": True,
            "num_rows": len(all_rows),
            "num_cols": max_cols
        }
    
    def process_image(
        self,
        image: Image.Image,
        **kwargs
    ) -> Tuple[str, Optional[Image.Image]]:
        """
        Main processing function - extracts tables using Docling.
        
        Args:
            image: Input PIL Image (not used directly, Docling reads from file)
            **kwargs: Additional arguments (ignored for Docling)
            
        Returns:
            Tuple of (text_result, result_image)
        """
        try:
            print("🏃 Running Docling extraction...")
            print(f"   Source: {self.source}")
            print(f"   Config: {self.source_config['description']}")
            
            # Extract tables using Docling
            tables = self.extract_tables()
            
            # Use the best (first valid) table or combine all
            if self.config['use_first_table_only']:
                valid_tables = [t for t in tables if t.get('valid', False)]
                table_data = valid_tables[0] if valid_tables else {
                    "columns": [], "data": [], "valid": False, 
                    "num_rows": 0, "num_cols": 0
                }
            else:
                # Combine all tables
                table_data = self._combine_tables(tables)
            
            # Calculate confidence
            confidence = self.calculate_confidence(tables if tables else [table_data])
            
            # Format as text output
            text_result = self._format_table_as_text(table_data, confidence)
            
            print(f"✅ Docling extraction complete.")
            print(f"   Tables found: {len(tables)}")
            print(f"   Confidence: {confidence:.3f}")
            
            return text_result, None  # Docling doesn't produce result images
            
        except Exception as e:
            error_msg = f"Docling extraction error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, None
    
    def _format_table_as_text(self, table_data: Dict, confidence: float) -> str:
        """Format extracted table data as text."""
        if not table_data.get('valid', False):
            return "No valid table structure found"
        
        lines = []
        lines.append(f"# Extracted Table (Confidence: {confidence:.3f})")
        lines.append(f"Dimensions: {table_data['num_rows']} rows × {table_data['num_cols']} columns")
        lines.append("")
        
        # Add table data
        for row in table_data['data']:
            row_text = " | ".join(str(cell) for cell in row)
            lines.append(row_text)
        
        return "\n".join(lines)