from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from termcolor import cprint

from src.evaluation.models.base_model import BaseModel
from src.evaluation.prompts import VISUAL_TABLE_QA_SYSTEM_PROMPT
from src.evaluation.config import MainConfig

class InternVL3Model(BaseModel):
    """
    InternVL3 model implementation for visual table QA.
    
    InternVL3 uses a message-based format with role/content structure.
    Supports both HF format (e.g., OpenGVLab/InternVL3-8B-hf) and 
    GitHub format (e.g., OpenGVLab/InternVL3-8B).
    
    Reference: https://huggingface.co/docs/transformers/en/model_doc/internvl
    """
    
    def __init__(self, cfg: MainConfig):
        super().__init__(cfg)

    def load_processor(self):
        cprint(f"Loading processor for: {self.cfg.model.model_path}", "blue")
        return AutoProcessor.from_pretrained(
            self.cfg.model.model_path, 
            trust_remote_code=True
        )

    def prepare_input(self, data_row: dict, images_dir: str) -> dict:
        """
        Prepares input for InternVL3 model.
        
        InternVL3 expects messages in the format:
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": <PIL.Image>},
                    {"type": "text", "text": "Question: ..."}
                ]
            }
        ]
        
        The processor.apply_chat_template handles the formatting.
        """
        table_id = data_row['table_id']
        image_type = self.cfg.dataset.image_type
        image_filename = data_row['image_filename']
        
        image_path = Path(images_dir) / table_id / image_type / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        resized_image = self._resize_image(image)

        # InternVL3 message format
        # Note: InternVL3 can accept a system message, but we'll include 
        # the system prompt in the user message for simplicity
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": f"{VISUAL_TABLE_QA_SYSTEM_PROMPT}\n\nQuestion: {data_row['question']}"}
                ]
            }
        ]
        
        # Apply chat template to format the prompt
        prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": resized_image}
        }