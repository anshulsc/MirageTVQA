from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from termcolor import cprint

from src.evaluation.models.base_model import BaseModel
from src.evaluation.prompts import VISUAL_TABLE_QA_SYSTEM_PROMPT
from src.evaluation.config import MainConfig

class LlamaVisionModel(BaseModel):
    """
    Llama 3.2 Vision model implementation for visual table QA.
    
    Supports Llama 3.2 Vision Instruct models (11B and 90B variants) which are
    the first Llama models with vision capabilities.
    
    Compatible models:
    - meta-llama/Llama-3.2-11B-Vision-Instruct
    - meta-llama/Llama-3.2-90B-Vision-Instruct
    
    These models use a message-based format with the special <|image|> token
    to indicate image placement in the prompt.
    
    Reference: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
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
        Prepares input for Llama 3.2 Vision models.
        
        Llama 3.2 Vision expects messages in the format:
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Question: ..."}
                ]
            }
        ]
        
        The processor.apply_chat_template handles the formatting and places
        the special <|image|> token where needed.
        """
        table_id = data_row['table_id']
        image_type = self.cfg.dataset.image_type
        image_filename = data_row['image_filename']
        
        image_path = Path(images_dir) / table_id / image_type / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        resized_image = self._resize_image(image)

        # Llama 3.2 Vision message format
        # The system prompt is included in the user message for simplicity
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{VISUAL_TABLE_QA_SYSTEM_PROMPT}\n\nQuestion: {data_row['question']}"}
                ]
            }
        ]
        
        # Apply chat template to format the prompt
        # This will insert the <|image|> token and proper message formatting
        prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # For vLLM, we pass the image separately in multi_modal_data
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": resized_image}
        }