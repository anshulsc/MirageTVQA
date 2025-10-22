from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from termcolor import cprint

from src.evaluation.models.base_model import BaseModel
from src.evaluation.prompts import VISUAL_TABLE_QA_SYSTEM_PROMPT
from src.evaluation.config import MainConfig

class QwenModel(BaseModel):
    
    def __init__(self, cfg: MainConfig):
        super().__init__(cfg)
        self.is_thinking_model = self._detect_thinking_model()
        self.enable_thinking = getattr(cfg.model, 'enable_thinking', None)
        
        if self.is_thinking_model:
            cprint(f"Initialized Qwen Thinking Model", "cyan")
            if self.enable_thinking is not None:
                cprint(f"Thinking mode: {'Enabled' if self.enable_thinking else 'Disabled'}", "cyan")
            else:
                cprint(f"Thinking mode: Default (model-dependent)", "cyan")

    def _detect_thinking_model(self) -> bool:
        model_path = self.cfg.model.model_path.lower()
        return 'thinking' in model_path or 'reasoning' in model_path

    def load_processor(self):
        cprint(f"Loading processor for: {self.cfg.model.model_path}", "blue")
        return AutoProcessor.from_pretrained(self.cfg.model.model_path, trust_remote_code=True)

    def prepare_input(self, data_row: dict, images_dir: str) -> dict:
        table_id = data_row['table_id']
        image_type = self.cfg.dataset.image_type
        image_filename = data_row['image_filename']
        
        image_path = Path(images_dir) / table_id / image_type / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        resized_image = self._resize_image(image)

        messages = [
            {"role": "system", "content": VISUAL_TABLE_QA_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"\nQuestion: {data_row['question']}"}
            ]},
        ]
        
        chat_template_kwargs = {}
        if self.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.enable_thinking
        
        if chat_template_kwargs:
            prompt_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                **chat_template_kwargs
            )
        else:
            prompt_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": resized_image}
        }
    
    def _parse_model_response(self, response_str: str) -> list:
        result = super()._parse_model_response(response_str)
        
        if self.is_thinking_model and '<think>' in response_str:
            try:
                think_start = response_str.find('<think>')
                think_end = response_str.find('</think>')
                
                if think_start != -1 and think_end != -1:
                    reasoning = response_str[think_start + 7:think_end].strip()
                    final_answer = response_str[think_end + 8:].strip()
                    
                    cprint(f"\n[Thinking Model] Reasoning length: {len(reasoning)} chars", "yellow")
                    
                    if final_answer:
                        try:
                            from src.evaluation.prompts import Response
                            from pydantic import ValidationError
                            import json
                            
                            parsed = Response.model_validate_json(final_answer)
                            return parsed.data
                        except (json.JSONDecodeError, ValidationError):
                            pass
            except Exception as e:
                cprint(f"[WARN] Error parsing thinking content: {e}", "yellow")
        
        return result