from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from termcolor import cprint

from src.evaluation.models.base_model import BaseModel
from src.evaluation.prompts import VISUAL_TABLE_QA_SYSTEM_PROMPT
from src.evaluation.config import MainConfig

class PangeaModel(BaseModel):
    """
    Pangea-7B model implementation for visual table QA.
    
    Pangea-7B is a fully open multilingual multimodal LLM supporting 39 languages.
    It follows the LLaVA-NeXT architecture with Qwen2-7B-Instruct as the backbone
    and clip-vit-large-patch14-336 as the vision encoder.
    
    Compatible models:
    - neulab/Pangea-7B (requires LLaVA-NeXT codebase)
    - neulab/Pangea-7B-hf (HuggingFace transformers compatible - recommended for vLLM)
    
    The model uses Qwen2's chat template format with special tokens:
    <|im_start|> and <|im_end|> for message boundaries.
    
    Note: Since Pangea uses the LLaVA-NeXT architecture and vLLM supports LLaVA-NeXT,
    this model should work with vLLM using the llava_next model class.
    
    Reference: 
    - https://huggingface.co/neulab/Pangea-7B
    - https://huggingface.co/neulab/Pangea-7B-hf
    - https://arxiv.org/abs/2410.16153
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
        Prepares input for Pangea-7B model.
        
        Pangea uses the Qwen2 chat template format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        <image>
        {user_message}<|im_end|>
        <|im_start|>assistant
        
        For vLLM with LLaVA-NeXT architecture, we use the standard message format
        and the processor handles the special tokens.
        """
        table_id = data_row['table_id']
        image_type = self.cfg.dataset.image_type
        image_filename = data_row['image_filename']
        
        image_path = Path(images_dir) / table_id / image_type / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        resized_image = self._resize_image(image)

        # Pangea uses Qwen2 chat format with system and user messages
        # The <image> token placement is important for LLaVA-NeXT architecture
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{VISUAL_TABLE_QA_SYSTEM_PROMPT}\n\nQuestion: {data_row['question']}"}
                ]
            }
        ]
        
        # Apply chat template to format the prompt
        # For Pangea-7B-hf, the processor handles the Qwen2 template formatting
        try:
            prompt_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback to manual formatting if apply_chat_template fails
            cprint(f"Warning: apply_chat_template failed, using manual formatting. Error: {e}", "yellow")
            prompt_text = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<image>\n{VISUAL_TABLE_QA_SYSTEM_PROMPT}\n\n"
                f"Question: {data_row['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": resized_image}
        }