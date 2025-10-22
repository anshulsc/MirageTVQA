import argparse
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "qwen"
    model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size: int = 2
    max_model_len: int = 65536
    gpu_memory_utilization: float = 0.90
    temperature: float = 0.0
    max_new_tokens: int = 1096
    batch_size: int = 4096

    enable_thinking: bool | None = None  
    enable_reasoning: bool = False 
    reasoning_parser: str = "qwen3" 

@dataclass
class DatasetConfig:
    name: str = "multitableqa"
    data_file: str = "/home/anshulsc/links/scratch/TableLingua/dataset_combined_final.jsonl"
    images_root_dir: str = "/home/anshulsc/links/scratch/TableLingua/images/"
    image_type: str = "noise"  
    lang_code: str = "default"     
    resolution: int | None = None

@dataclass
class PromptConfig:
    name: str = "default_visual_qa"

@dataclass
class MainConfig:
  
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output_file_template: str = "data/processed/evaluation_results_new/{model_name}_{dataset_name}_{image_type}"
    no_batch: bool = False
    resume_from: str | None = None 
    
def get_config() -> MainConfig:
    parser = argparse.ArgumentParser(description="Run VLLM evaluation for MultiTableQA with thinking model support.")
    
    parser.add_argument("--model_name", type=str, help="Name of the model to use (e.g., qwen, gemma).")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--batch_size", type=int, help="Batch size for inference (default: 8).")
    
    parser.add_argument(
        "--enable_thinking", 
        type=str, 
        choices=['true', 'false', 'default'],
        default='default',
        help="Enable/disable thinking mode for Qwen3 thinking models. 'default' uses model's default behavior."
    )
    parser.add_argument(
        "--enable_reasoning",
        action='store_true',
        help="Enable reasoning mode in vLLM (for thinking models)."
    )
    parser.add_argument(
        "--reasoning_parser",
        type=str,
        default="deepseek_r1",
        help="Reasoning parser to use (default: deepseek_r1 for Qwen3)."
    )
    
    parser.add_argument("--data_file", type=str, help="Path to the .jsonl data file.")
    parser.add_argument("--images_root_dir", type=str, help="Path to the root directory of images.")
    parser.add_argument("--image_type", type=str, choices=['clean', 'noise'], help="Image type to evaluate.")
    parser.add_argument("--lang_code", type=str, help="Language code to filter by (e.g., 'en', 'es', 'default').")
    
    parser.add_argument("--no_batch", action='store_true', help="Disable batch processing (use single-item mode).")
    parser.add_argument("--resume_from", type=str, help="Path to incomplete evaluation jsonl file to resume from.")
    
    args = parser.parse_args()
    
    cfg = MainConfig()
    
    if args.model_name:
        cfg.model.name = args.model_name
    if args.model_path:
        cfg.model.model_path = args.model_path
    if args.batch_size:
        cfg.model.batch_size = args.batch_size
    
    # Handle thinking model configuration
    if args.enable_thinking == 'true':
        cfg.model.enable_thinking = True
    elif args.enable_thinking == 'false':
        cfg.model.enable_thinking = False
    # 'default' keeps it as None
    
    if args.enable_reasoning:
        cfg.model.enable_reasoning = True
    
    if args.reasoning_parser:
        cfg.model.reasoning_parser = args.reasoning_parser
    
    # Dataset arguments
    if args.data_file:
        cfg.dataset.data_file = args.data_file
    if args.images_root_dir:
        cfg.dataset.images_root_dir = args.images_root_dir
    if args.image_type:
        cfg.dataset.image_type = args.image_type
    if args.lang_code:
        cfg.dataset.lang_code = args.lang_code
    
    if args.no_batch:
        cfg.no_batch = True
    if args.resume_from:
        cfg.resume_from = args.resume_from
        
    return cfg