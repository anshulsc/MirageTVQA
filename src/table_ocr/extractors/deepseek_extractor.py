"""
DeepSeek-OCR Extractor
Handles model inference, bounding box detection, and result formatting
"""
import re
import os
import tempfile
from typing import Optional, Tuple, List

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer

from src.configs import text_extraction_config as cfg
from src.table_ocr.extractors.base_extractor import BaseExtractor


class DeepSeekExtractor(BaseExtractor):
    """DeepSeek-OCR implementation of the base extractor."""
    
    # Class-level model cache (shared within same process)
    _shared_model = None
    _shared_tokenizer = None
    _model_loaded = False
    
    def __init__(self, image_path, source='default', metadata=None):
        super().__init__(image_path, source, metadata)
        
        # Get DeepSeek config
        self.config = cfg.DEEPSEEK_CONFIG
        
        # Compile regex pattern for bounding box detection
        self.bbox_pattern = re.compile(
            r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>"
        )
        
        # Load model if not already loaded
        self._ensure_model_loaded()
    
    @classmethod
    def _ensure_model_loaded(cls):
        """Load model and tokenizer once per process."""
        if cls._model_loaded:
            return
        
        print("=" * 70)
        print("Loading DeepSeek-OCR model and tokenizer...")
        print(f"Model: {cfg.DEEPSEEK_CONFIG['model_name']}")
        print(f"Process ID: {os.getpid()}")
        print("=" * 70)
        
        try:
            # Load tokenizer
            cls._shared_tokenizer = AutoTokenizer.from_pretrained(
                cfg.DEEPSEEK_CONFIG['model_name'],
                trust_remote_code=True
            )
            
            # Load model to CPU initially
            cls._shared_model = AutoModel.from_pretrained(
                cfg.DEEPSEEK_CONFIG['model_name'],
                _attn_implementation=cfg.DEEPSEEK_CONFIG['model_implementation'],
                trust_remote_code=True,
                use_safetensors=cfg.DEEPSEEK_CONFIG['use_safetensors']
            )
            cls._shared_model = cls._shared_model.eval()
            
            cls._model_loaded = True
            print("✅ Model loaded successfully.\n")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def build_prompt(self, task_type: str, ref_text: Optional[str] = None) -> str:
        """Build inference prompt based on task type."""
        task_config = self.config['task_configs'].get(task_type)
        
        if not task_config:
            # Fallback to Free OCR
            task_config = self.config['task_configs']["📝 Free OCR"]
        
        prompt_template = task_config["prompt_template"]
        
        # Handle reference text for locate tasks
        if task_config["requires_ref"]:
            if not ref_text or not ref_text.strip():
                raise ValueError(cfg.ERROR_MESSAGES["missing_ref"])
            return prompt_template.format(ref_text=ref_text.strip())
        
        return prompt_template
    
    def extract_bounding_boxes(self, text_result: str) -> List[Tuple[int, int, int, int]]:
        """Extract all bounding box coordinates from text result."""
        matches = list(self.bbox_pattern.finditer(text_result))
        
        bboxes = []
        for match in matches:
            coords = [int(c) for c in match.groups()]
            bboxes.append(tuple(coords))
        
        return bboxes
    
    def draw_bounding_boxes(
        self, 
        image: Image.Image, 
        bboxes: List[Tuple[int, int, int, int]]
    ) -> Image.Image:
        """Draw bounding boxes on image."""
        if not bboxes:
            return None
        
        image_with_bboxes = image.copy()
        draw = ImageDraw.Draw(image_with_bboxes)
        width, height = image.size
        
        for x1_norm, y1_norm, x2_norm, y2_norm in bboxes:
            # Scale normalized coordinates to actual image size
            x1 = int(x1_norm / self.config['bbox_coordinate_space'] * width)
            y1 = int(y1_norm / self.config['bbox_coordinate_space'] * height)
            x2 = int(x2_norm / self.config['bbox_coordinate_space'] * width)
            y2 = int(y2_norm / self.config['bbox_coordinate_space'] * height)
            
            # Draw rectangle
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=self.config['bbox_color'],
                width=self.config['bbox_width']
            )
        
        return image_with_bboxes
    
    def find_result_image(self, output_path: str) -> Optional[Image.Image]:
        """Find pre-generated result image in output directory."""
        try:
            for filename in os.listdir(output_path):
                if "grounding" in filename or "result" in filename:
                    image_path = os.path.join(output_path, filename)
                    return Image.open(image_path)
        except Exception as e:
            print(f"⚠️ Error loading result image: {e}")
        
        return None
    
    def process_image(
        self,
        image: Image.Image,
        model_size: Optional[str] = None,
        task_type: Optional[str] = None,
        ref_text: Optional[str] = None
    ) -> Tuple[str, Optional[Image.Image]]:
        """
        Main processing function - runs OCR inference and generates results.
        
        Args:
            image: Input PIL Image
            model_size: Resolution configuration key
            task_type: OCR task type
            ref_text: Optional reference text for locate tasks
            
        Returns:
            Tuple of (text_result, result_image)
        """
        # Use defaults if not provided
        model_size = model_size or self.config['default_size']
        task_type = task_type or self.config['default_task']
        
        try:
            # Move model to GPU
            print("🚀 Moving model to GPU...")
            model_gpu = self._shared_model.cuda().to(torch.bfloat16)
            print("✅ Model is on GPU.")
            
            with tempfile.TemporaryDirectory() as output_path:
                # Build prompt
                try:
                    prompt = self.build_prompt(task_type, ref_text)
                except ValueError as e:
                    return str(e), None
                
                # Save temporary image
                temp_image_path = os.path.join(output_path, "temp_image.png")
                image.save(temp_image_path)
                
                # Get size configuration
                size_config = self.config['size_configs'].get(
                    model_size,
                    self.config['size_configs'][self.config['default_size']]
                )
                
                print(f"🏃 Running DeepSeek inference...")
                print(f"   Task: {task_type}")
                print(f"   Size: {model_size}")
                
                # Run inference
                text_result = model_gpu.infer(
                    self._shared_tokenizer,
                    prompt=prompt,
                    image_file=temp_image_path,
                    output_path=output_path,
                    base_size=size_config["base_size"],
                    image_size=size_config["image_size"],
                    crop_mode=size_config["crop_mode"],
                    save_results=self.config['save_results'],
                    test_compress=self.config['test_compress'],
                    eval_mode=self.config['eval_mode']
                )
                
                print(f"✅ Inference complete.")
                
                # Process bounding boxes
                result_image = self._process_bounding_boxes(
                    text_result,
                    image,
                    output_path
                )
                
                return text_result, result_image
                
        except torch.cuda.OutOfMemoryError:
            error_msg = cfg.ERROR_MESSAGES["gpu_oom"]
            print(f"❌ {error_msg}")
            return error_msg, None
            
        except Exception as e:
            error_msg = cfg.ERROR_MESSAGES["model_error"].format(error=str(e))
            print(f"❌ {error_msg}")
            return error_msg, None
            
        finally:
            # Cleanup GPU memory
            if self.config['gpu_memory_cleanup']:
                torch.cuda.empty_cache()
    
    def _process_bounding_boxes(
        self,
        text_result: str,
        original_image: Image.Image,
        output_path: str
    ) -> Optional[Image.Image]:
        """Process and visualize bounding boxes from OCR output."""
        # Try to extract bounding boxes from text
        bboxes = self.extract_bounding_boxes(text_result)
        
        if bboxes:
            print(f"✅ Found {len(bboxes)} bounding box(es). Drawing...")
            return self.draw_bounding_boxes(original_image, bboxes)
        
        # Fallback: look for pre-generated result image
        print("ℹ️ No bounding boxes in text. Searching for result image...")
        return self.find_result_image(output_path)