from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from termcolor import cprint
import os

from src.evaluation.models.base_model import BaseModel
from src.evaluation.prompts import VISUAL_TABLE_QA_SYSTEM_PROMPT
from src.evaluation.config import MainConfig

class Phi4MultimodalModel(BaseModel):
    """
    Phi-4-multimodal-instruct model implementation for visual table QA.
    
    Phi-4-multimodal-instruct is a lightweight multimodal model (5.6B parameters) 
    that processes text, image, and audio inputs. It has 128K context length and 
    uses Phi-4-Mini-Instruct as the backbone with advanced vision and speech encoders.
    
    Compatible model:
    - microsoft/Phi-4-multimodal-instruct
    
    The model uses special tokens for different modalities:
    - <|image_1|>, <|image_2|>, ... for images
    - <|audio_1|>, <|audio_2|>, ... for audio
    - <|system|>, <|user|>, <|assistant|>, <|end|> for chat structure
    
    **IMPORTANT: vLLM Setup**
    Phi-4-multimodal-instruct requires LoRA adapters for vision and speech modalities.
    When using vLLM, you need to:
    1. Download the model which includes speech-lora and vision-lora folders
    2. Pass these folders to vLLM via --lora-modules argument
    3. Enable LoRA support with appropriate settings
    
    For offline inference with vLLM (as used in this BaseModel), the LLM is initialized
    with the base model path, and LoRA adapters should be loaded separately if needed.
    
    Reference: 
    - https://huggingface.co/microsoft/Phi-4-multimodal-instruct
    - https://arxiv.org/abs/2503.01743
    """
    
    def __init__(self, cfg: MainConfig):
        super().__init__(cfg)
        # Get LoRA paths for vision and speech if available
        self.vision_lora_path = self._get_lora_path("vision-lora")
        self.speech_lora_path = self._get_lora_path("speech-lora")
        
        if self.vision_lora_path:
            cprint(f"Found vision LoRA at: {self.vision_lora_path}", "green")
        else:
            cprint("Warning: vision-lora not found. Model may not work properly for vision tasks.", "yellow")

    def _get_lora_path(self, lora_subfolder: str) -> str:
        """
        Get the path to LoRA adapters from the model directory.
        For vLLM, these are typically in the cached model directory.
        """
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id=self.cfg.model.model_path)
            lora_path = os.path.join(model_path, lora_subfolder)
            if os.path.exists(lora_path):
                return lora_path
        except Exception as e:
            cprint(f"Could not locate {lora_subfolder}: {e}", "yellow")
        return None

    def load_processor(self):
        cprint(f"Loading processor for: {self.cfg.model.model_path}", "blue")
        return AutoProcessor.from_pretrained(
            self.cfg.model.model_path, 
            trust_remote_code=True
        )

    def prepare_input(self, data_row: dict, images_dir: str) -> dict:
        """
        Prepares input for Phi-4-multimodal-instruct model.
        
        Phi-4 uses a specific chat format:
        <|system|>You are a helpful assistant.<|end|>
        <|user|><|image_1|>Question text<|end|>
        <|assistant|>
        
        For vision tasks, use <|image_1|>, <|image_2|>, etc. placeholders.
        For audio tasks, use <|audio_1|>, <|audio_2|>, etc. placeholders.
        
        The processor handles the formatting, but for vLLM we need to construct
        the prompt manually with the correct special tokens.
        """
        table_id = data_row['table_id']
        image_type = self.cfg.dataset.image_type
        image_filename = data_row['image_filename']
        
        image_path = Path(images_dir) / table_id / image_type / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        resized_image = self._resize_image(image)

        # Phi-4-multimodal chat format
        # System message + User message with image placeholder
        system_message = "You are a helpful assistant."
        user_prompt = f"<|image_1|>{VISUAL_TABLE_QA_SYSTEM_PROMPT}\n\nQuestion: {data_row['question']}"
        
        # Construct the full prompt with proper special tokens
        prompt = (
            f"<|system|>{system_message}<|end|>"
            f"<|user|>{user_prompt}<|end|>"
            f"<|assistant|>"
        )

        # For vLLM with multimodal support, we pass the image separately
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": resized_image}
        }