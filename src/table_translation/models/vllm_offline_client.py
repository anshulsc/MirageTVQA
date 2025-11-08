import json
import re
from termcolor import cprint
from typing import List, Dict
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer

from .base_client import BaseModelClient
from ..prompts import TableJSON


class VLLMOfflineClient(BaseModelClient):

    def __init__(
        self, 
        model_name: str, 
        tensor_parallel_size: int = 1, 
        gpu_memory_utilization: float = 0.9,
        is_thinking_model: bool = False
    ):
        cprint(f"Loading vLLM offline engine for: {model_name}", "yellow")
        cprint(f"  Tensor Parallel Size: {tensor_parallel_size}", "yellow")
        cprint(f"  GPU Memory Utilization: {gpu_memory_utilization}", "yellow")
        cprint(f"  Thinking Model: {is_thinking_model}", "yellow")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=32768,  
        )
        self.model_name = model_name
        self.is_thinking_model = is_thinking_model
        
        json_schema = TableJSON.model_json_schema() 

        if self.is_thinking_model:
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=8192,  
                stop=None,
            )
            cprint(f"  Using thinking mode: Will extract JSON after </think> tag", "cyan")
        else:
            # For non-thinking models, use guided decoding directly
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=4096,
                stop=None,
                guided_decoding=GuidedDecodingParams(json=json_schema)
            )
            cprint(f"  Using direct JSON generation mode", "cyan")
        
        cprint(f"✓ vLLM Offline Client initialized for: {model_name}", "green")

    def _extract_json_from_thinking_output(self, text: str) -> str:
        
        think_end_pattern = r'</think>\s*({.*})\s*$'
        match = re.search(think_end_pattern, text, re.DOTALL)
        
        if match:
            json_content = match.group(1)
            return json_content
        
        json_pattern = r'({[\s\S]*})'
        matches = re.findall(json_pattern, text)
        
        if matches:
            for json_candidate in reversed(matches):
                try:
                    json.loads(json_candidate)
                    return json_candidate
                except:
                    continue
        return text

    def generate_structured_json(self, prompt: str, max_retries: int = 3) -> TableJSON | None:
        results = self.generate_structured_json_batch([{"id": "single", "prompt": prompt}])
        return results.get("single")

    def generate_structured_json_batch(
        self, 
        prompts: List[Dict[str, str]], 
        max_retries: int = 2
    ) -> Dict[str, TableJSON | None]:
        results = {}
        
        for attempt in range(max_retries):
            
            tokenizer = self.llm.get_tokenizer()
            remaining_prompts_with_templated_text = []
            for p in prompts:
                if p["id"] not in results or results[p["id"]] is None:
                    messages_for_prompt = [{"role": "user", "content": p["prompt"]}]
                    
                  
                    if self.is_thinking_model:
                        templated_text = tokenizer.apply_chat_template(
                            messages_for_prompt,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True, 
                        )
                    else:
                        templated_text = tokenizer.apply_chat_template(
                            messages_for_prompt,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    
                    remaining_prompts_with_templated_text.append({"id": p["id"], "prompt": templated_text})
            
            remaining_prompts = remaining_prompts_with_templated_text
            
            if not remaining_prompts:
                break
            
            if self.is_thinking_model:
                cprint(f"\nBatch inference (THINKING MODE): {len(remaining_prompts)} prompts (attempt {attempt + 1})...", "magenta")
            else:
                cprint(f"\nBatch inference: {len(remaining_prompts)} prompts (attempt {attempt + 1})...", "cyan")
            
            try:
                prompt_texts = [p["prompt"] for p in remaining_prompts]
                prompt_ids = [p["id"] for p in remaining_prompts]

                outputs = self.llm.generate(prompt_texts, self.sampling_params)
   
                for i, output in enumerate(outputs):
                    prompt_id = prompt_ids[i]
                    generated_text = output.outputs[0].text
                    
                    try:
                        if self.is_thinking_model:
                            json_content = self._extract_json_from_thinking_output(generated_text)
                            
                            if '</think>' in generated_text:
                                thinking_part = generated_text.split('</think>')[0]
                                cprint(f"{prompt_id}: Thinking length = {len(thinking_part)} chars", "cyan")
                            
                            table_json = TableJSON.model_validate_json(json_content)
                        else:
                            table_json = TableJSON.model_validate_json(generated_text)
                        
                        results[prompt_id] = table_json
                        cprint(f"  ✓ {prompt_id}: Success", "green")
                    except Exception as e:
                        cprint(f"  ✗ {prompt_id}: Parse failed - {str(e)[:100]}", "yellow")
                        if self.is_thinking_model:
                            snippet = generated_text[-200:] if len(generated_text) > 200 else generated_text
                            cprint(f"    Output snippet: ...{snippet}", "red")
                        else:
                            cprint(generated_text[:500], "red")
                        results[prompt_id] = None
                        
            except Exception as e:
                cprint(f"Batch inference failed: {e}", "red")
                import traceback
                traceback.print_exc()
                for p in remaining_prompts:
                    if p["id"] not in results:
                        results[p["id"]] = None
        
        successful = sum(1 for r in results.values() if isinstance(r, TableJSON))
        cprint(f"\n✓ Batch complete: {successful}/{len(prompts)} successful", "green" if successful == len(prompts) else "yellow")
        
        failed = [pid for pid, result in results.items() if result is None or not isinstance(result, TableJSON)]
        if failed:
            cprint(f"✗ Failed IDs: {', '.join(failed)}", "red")
            
        return results

    def __del__(self):
        if hasattr(self, 'llm'):
            del self.llm
            cprint("vLLM offline engine cleaned up", "yellow")