from pathlib import Path
from PIL import Image, ImageStat
from transformers import AutoProcessor
from termcolor import cprint

from src.evaluation.models.base_model import BaseModel
from src.evaluation.prompts import VISUAL_TABLE_QA_SYSTEM_PROMPT
from src.evaluation.config import MainConfig

class MolmoModel(BaseModel):
    
    def __init__(self, cfg: MainConfig):
        super().__init__(cfg)

    def load_processor(self):
        cprint(f"Loading processor for: {self.cfg.model.model_path}", "blue")
        return AutoProcessor.from_pretrained(
            self.cfg.model.model_path, 
            trust_remote_code=True
        )

    def _add_background_to_image(self, image: Image.Image) -> Image.Image:

        if image.mode != 'RGBA':
            return image
        
        gray_image = image.convert('L')
        stat = ImageStat.Stat(gray_image)
        average_brightness = stat.mean[0]
        
        # Define background color based on brightness
        bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)
        
        # Create new image with background
        new_image = Image.new('RGB', image.size, bg_color)
        new_image.paste(image, (0, 0), image)
        
        return new_image

    def prepare_input(self, data_row: dict, images_dir: str) -> dict:

        table_id = data_row['table_id']
        image_type = self.cfg.dataset.image_type
        image_filename = data_row['image_filename']
        
        image_path = Path(images_dir) / table_id / image_type / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load and preprocess image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            # Handle transparent images by adding background
            if image.mode == "RGBA":
                image = self._add_background_to_image(image)
            else:
                image = image.convert("RGB")
        
        resized_image = self._resize_image(image)

        # Molmo uses a simple text prompt format
        # We combine the system prompt and question
        prompt_text = f"{VISUAL_TABLE_QA_SYSTEM_PROMPT}\n\nQuestion: {data_row['question']}"

        # For vLLM with Molmo, we pass the prompt and image
        # The processor will handle the formatting internally
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": resized_image}
        }