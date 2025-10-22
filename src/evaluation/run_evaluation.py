import os
from datetime import datetime
from pathlib import Path
from termcolor import cprint

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'

from .config import get_config
from src.evaluation.models.qwen import QwenModel
from src.evaluation.models.gemma3 import Gemma3Model
from src.evaluation.models.internVL3 import InternVL3Model
from src.evaluation.models.llama import LlamaVisionModel
from src.evaluation.models.pangea import PangeaModel
from src.evaluation.models.phi4 import Phi4MultimodalModel
from src.evaluation.models.molmo import MolmoModel

from src.evaluation.data_loader import load_benchmark_data

MODEL_EVALUATOR_MAPPING = {
    "qwen": QwenModel,
    "gemma3": Gemma3Model,
    "internVL": InternVL3Model,
    "llama": LlamaVisionModel,
    "pangea": PangeaModel,
    "phi4": Phi4MultimodalModel,
    "molmo": MolmoModel
}

def get_output_filepath(cfg) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = cfg.model.model_path.replace("/", "_")
    lang_str = cfg.dataset.lang_code
    
    output_file_base = cfg.output_file_template.format(
        model_name=model_name_safe,
        dataset_name=cfg.dataset.name,
        image_type=cfg.dataset.image_type
    )

    return f"{output_file_base}_{lang_str}_{timestamp}.jsonl"

def main():
    cfg = get_config()

    cprint("--- MultiTableQA Evaluation Pipeline (Python Config) ---", "magenta", attrs=["bold"])
    cprint("Running with the following configuration:", "yellow")
    cprint(f"  - Model: {cfg.model.name} ({cfg.model.model_path})", "yellow")
    cprint(f"  - Dataset: {cfg.dataset.data_file}", "yellow")
    cprint(f"  - Images: {cfg.dataset.image_type} (lang: {cfg.dataset.lang_code})", "yellow")
    cprint(f"  - Batch size: {cfg.model.batch_size}", "yellow")
    if cfg.resume_from:
        cprint(f"  - Resuming from: {cfg.resume_from}", "yellow", attrs=["bold"])
    cprint("-" * 50, "magenta")

    model_name = cfg.model.name
    model_class = MODEL_EVALUATOR_MAPPING.get(model_name)
    if model_class is None:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(MODEL_EVALUATOR_MAPPING.keys())}")

    try:
        # 1. Load Data (with resume functionality)
        data = load_benchmark_data(
            data_file_path=cfg.dataset.data_file,
            images_root_dir=cfg.dataset.images_root_dir,
            image_type=cfg.dataset.image_type,
            lang_code_filter=cfg.dataset.lang_code,
            resume_from=cfg.resume_from
        )
        if not data:
            cprint("No data loaded for the specified criteria. All instances may be completed already.", "yellow")
            return

        model = model_class(cfg)

        # Determine output file
        if cfg.resume_from:
            # Continue writing to the same file
            output_file = cfg.resume_from
            cprint(f"Appending results to: {output_file}", "cyan", attrs=["bold"])
        else:
            output_file = get_output_filepath(cfg)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            cprint(f"Output will be saved to: {output_file}", "cyan")

        import sys
        use_batch = True
        if hasattr(cfg, 'no_batch'):
            use_batch = not cfg.no_batch
        elif '--no_batch' in sys.argv:
            use_batch = False
        
        model.evaluate(data, output_file, cfg.dataset.images_root_dir, use_batch=use_batch)
        cprint(f"\nEvaluation finished successfully!", "green", attrs=["bold"])
        cprint(f"Results saved to {output_file}", "green")

    except Exception as e:
        cprint(f"\nFATAL ERROR during execution: {str(e)}", "red", attrs=["bold"])
        raise

if __name__ == "__main__":
    main()