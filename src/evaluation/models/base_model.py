import json
import signal
import torch
from pathlib import Path
from PIL import Image
from termcolor import cprint
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import ValidationError

from src.evaluation.prompts import Response
from src.evaluation.config import MainConfig

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Inference timed out!")

class BaseModel:
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        self.model = self.load_model()
        self.processor = self.load_processor()
        self.resolution = cfg.dataset.resolution
        self.batch_size = getattr(cfg.model, 'batch_size', 8)

    def load_model(self):
        cprint(f"Loading VLLM engine for model: {self.cfg.model.model_path}", "yellow")
        cprint("This may take a few minutes...", "yellow")
        
        llm_params = {
            "model": self.cfg.model.model_path,
            "tensor_parallel_size": self.cfg.model.tensor_parallel_size,
            "max_model_len": self.cfg.model.max_model_len,
            "gpu_memory_utilization": self.cfg.model.gpu_memory_utilization,
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"image": 10, "video": 0}
        }
        
        if hasattr(self.cfg.model, 'enable_reasoning') and self.cfg.model.enable_reasoning:
            cprint(f"Enabling reasoning mode with parser: {self.cfg.model.reasoning_parser}", "cyan")
        
        llm = LLM(**llm_params)
        cprint("VLLM Engine loaded successfully.", "green")
        return llm

    def load_processor(self):
        raise NotImplementedError("Each model must implement its own processor loader.")

    def prepare_input(self, data_row, images_dir):
        raise NotImplementedError("Each model must implement its own input preparer.")

    def _create_vllm_sampling_params(self) -> SamplingParams:
        try:
            json_schema = Response.model_json_schema()
            guided_params = GuidedDecodingParams(json=json_schema)
        except Exception as e:
            cprint(f"Warning: Could not create guided decoding params. Error: {e}", "red")
            guided_params = None

        sampling_params = SamplingParams(
            temperature=self.cfg.model.temperature,
            max_tokens=self.cfg.model.max_new_tokens,
            guided_decoding=guided_params,
        )
        
        return sampling_params

    def generate_response(self, inputs: dict) -> str:
        sampling_params = self._create_vllm_sampling_params()
        
        request = {
            "prompt": inputs["prompt"],
            "multi_modal_data": inputs.get("multi_modal_data")
        }
        
        if "chat_template_kwargs" in inputs:
            pass
        
        outputs = self.model.generate(request, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch_responses(self, batch_inputs: list) -> list:
        sampling_params = self._create_vllm_sampling_params()
        
        requests = []
        for inp in batch_inputs:
            request = {
                "prompt": inp["prompt"],
                "multi_modal_data": inp.get("multi_modal_data")
            }
            requests.append(request)
        
        try:
            outputs = self.model.generate(requests, sampling_params=sampling_params)
            return [output.outputs[0].text for output in outputs]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                cprint("GPU OOM during batch processing. Falling back to single-item mode.", "red")
                raise
            raise

    def _parse_model_response(self, response_str: str) -> list:

        try:
            parsed_response = Response.model_validate_json(response_str)
            return parsed_response.data
        except (json.JSONDecodeError, ValidationError) as e:
            if '<think>' in response_str and '</think>' in response_str:
                try:
                    think_end = response_str.rfind('</think>')
                    final_content = response_str[think_end + 8:].strip()
                    
                    if final_content:
                        parsed_response = Response.model_validate_json(final_content)
                        return parsed_response.data
                except (json.JSONDecodeError, ValidationError):
                    pass
            
            cprint(f"\n[WARN] Failed to parse model output as valid JSON: {response_str[:200]}... Error: {e}", "yellow")
            return [[response_str]]
        except Exception as e:
            cprint(f"\n[WARN] An unexpected error occurred during parsing: {e}", "yellow")
            return [["UNEXPECTED_PARSING_ERROR", str(e)]]

    def evaluate(self, data: list, output_file: str, images_dir: str, use_batch: bool = True):
        cprint(f"Starting evaluation on {len(data)} instances...", "cyan")
        
        if use_batch:
            cprint(f"Using batch processing with batch_size={self.batch_size}", "green")
            self._evaluate_batch(data, output_file, images_dir)
        else:
            cprint("Using single-item processing", "yellow")
            self._evaluate_single(data, output_file, images_dir)

    def _evaluate_single(self, data: list, output_file: str, images_dir: str):

        with open(output_file, "a+", encoding="utf-8") as out_file:
            for i, row in enumerate(tqdm(data, desc="Evaluating")):
                try:
                    inputs = self.prepare_input(row, images_dir)
                    raw_response_str = self.generate_response_with_timeout(inputs, timeout=300)
                    parsed_response_data = self._parse_model_response(raw_response_str)
                    result = self.create_result_dict(row, parsed_response_data)
                except Exception as e:
                    result = self.handle_exception(row, e)
                
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                if (i + 1) % 20 == 0:
                    out_file.flush()

    def _evaluate_batch(self, data: list, output_file: str, images_dir: str):
        """Batch evaluation for improved throughput."""
        with open(output_file, "a+", encoding="utf-8") as out_file:
            for batch_start in tqdm(range(0, len(data), self.batch_size), desc="Evaluating batches"):
                batch_end = min(batch_start + self.batch_size, len(data))
                batch_data = data[batch_start:batch_end]
                
                # Prepare inputs for the entire batch
                batch_inputs = []
                batch_rows = []
                failed_indices = []
                
                for idx, row in enumerate(batch_data):
                    try:
                        inputs = self.prepare_input(row, images_dir)
                        batch_inputs.append(inputs)
                        batch_rows.append(row)
                    except Exception as e:
                        cprint(f"\nError preparing input for question {row['question_id']}: {e}", "red")
                        result = self.handle_exception(row, e)
                        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        failed_indices.append(idx)
                
                if not batch_inputs:
                    continue
                
                try:
                    raw_responses = self.generate_batch_responses_with_timeout(batch_inputs, timeout=300)
                    
                    for row, raw_response in zip(batch_rows, raw_responses):
                        try:
                            parsed_response_data = self._parse_model_response(raw_response)
                            result = self.create_result_dict(row, parsed_response_data)
                        except Exception as e:
                            result = self.handle_exception(row, e)
                        
                        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
                except TimeoutException as e:
                    cprint(f"\nBatch inference timed out: {e}", "red")
                    for row in batch_rows:
                        result = self.handle_exception(row, e)
                        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as e:
                    cprint(f"\nBatch generation error: {e}", "red")
                    for row in batch_rows:
                        result = self.handle_exception(row, e)
                        out_file.write(json.dumps(result,ensure_ascii=False) + "\n")
                
                if (batch_end) % 100 == 0:
                    out_file.flush()

    def generate_response_with_timeout(self, inputs, timeout=120):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = self.generate_response(inputs)
        except TimeoutException:
            raise TimeoutException(f"Inference timed out after {timeout} seconds")
        finally:
            signal.alarm(0)
        return result

    def generate_batch_responses_with_timeout(self, batch_inputs, timeout=300):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = self.generate_batch_responses(batch_inputs)
        except TimeoutException:
            raise TimeoutException(f"Batch inference timed out after {timeout} seconds")
        finally:
            signal.alarm(0)
        return result

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if self.resolution and isinstance(self.resolution, int):
            return image.resize((self.resolution, self.resolution), Image.LANCZOS)
        return image

    def create_result_dict(self, row, parsed_data: list):
        return {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "golden_answer": row.get("golden_answer"),
            "image_filename": row.get("image_filename"),
            "reasoning_category": row.get("reasoning_category"),
            "model_response": parsed_data,
            "model_name": self.cfg.model.model_path,
        }

    def handle_exception(self, row, e):
        """Handle exceptions during inference."""
        cprint(f"\nError processing question {row['question_id']}: {e}", "red")
        error_message = f"Error: {e}"
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            error_message = "CUDA out of memory. Skipping."
        elif isinstance(e, TimeoutException):
            error_message = "Inference timed out."
        return self.create_result_dict(row, [["GENERATION_ERROR", error_message]])