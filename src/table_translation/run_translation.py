import time
from tqdm import tqdm
from pathlib import Path
from termcolor import cprint

from src.table_translation.models import vLLM_client
from src.configs import translation_config as cfg
from src.table_translation.models.gemini_client import GeminiClient
from src.table_translation.translator import TableTranslator


VLLM_MODE = getattr(cfg, 'VLLM_MODE', 'api')  

if VLLM_MODE == 'offline':
    from src.table_translation.models.vllm_offline_client import VLLMOfflineClient as VLLMClient
    cprint("Using vLLM OFFLINE mode (native batch inference)", "magenta", attrs=["bold"])
else:
    from src.table_translation.models.vLLM_client import VLLMClient
    cprint("Using vLLM API mode (OpenAI-compatible)", "magenta", attrs=["bold"])


def check_translation_completion(table_id: str, languages: dict) -> bool:
    table_output_dir = cfg.TRANSLATED_TABLES_DIR / table_id
    if not table_output_dir.is_dir():
        return False

    for lang_code in languages.keys():
        translated_file_path = table_output_dir / f"{lang_code}.json"
        if not translated_file_path.is_file():
            return False
    return True


def get_active_pipeline_steps():
    if hasattr(cfg, 'PIPELINE_MODE') and cfg.PIPELINE_MODE in cfg.PIPELINE_MODES:
        return cfg.PIPELINE_MODES[cfg.PIPELINE_MODE]
    elif hasattr(cfg, 'PIPELINE_STEPS'):
        return cfg.PIPELINE_STEPS
    else:
        return {
            "initial_translation": True,
            "refinement": True,
            "back_translation": True,
            "evaluation": True,
        }


def print_pipeline_config(steps):
    print("\n" + "="*70)
    print("  PIPELINE CONFIGURATION")
    print("="*70)
    print(f"  Step 1 - Initial Translation:  {'✓ ENABLED' if steps['initial_translation'] else '✗ SKIPPED'}")
    print(f"  Step 2 - Refinement:            {'✓ ENABLED' if steps['refinement'] else '✗ SKIPPED'}")
    print(f"  Step 3 - Back-Translation:      {'✓ ENABLED' if steps['back_translation'] else '✗ SKIPPED'}")
    print(f"  Step 4 - Evaluation & Saving:   {'✓ ENABLED' if steps['evaluation'] else '✗ SKIPPED'}")
    print("="*70)


def main():
    start_time = time.time()
    pipeline_steps = get_active_pipeline_steps()
    
    print("\n" + "="*70)
    print("  STARTING PHASE 1.2: TABLE TRANSLATION PIPELINE")
    if hasattr(cfg, 'PIPELINE_MODE'):
        print(f"              MODE: {cfg.PIPELINE_MODE.upper()}")
    if getattr(cfg, 'VLLM_IS_THINKING_MODEL', False):
        print("THINKING MODEL ENABLED")
    print("="*70)
    

    print_pipeline_config(pipeline_steps)
    vllm_client = None
    gemini_client = None
    
    if pipeline_steps['initial_translation']:
        if VLLM_MODE == 'offline':
            vllm_client = VLLMClient(
                model_name=cfg.VLLM_MODEL_NAME,
                tensor_parallel_size=getattr(cfg, 'VLLM_TENSOR_PARALLEL_SIZE', 1),
                gpu_memory_utilization=getattr(cfg, 'VLLM_GPU_MEMORY_UTIL', 0.9),
                is_thinking_model=getattr(cfg, 'VLLM_IS_THINKING_MODEL', False)
            )
        else:
            vllm_client = VLLMClient(
                base_url=cfg.VLLM_BASE_URL,
                api_key=cfg.VLLM_API_KEY,
                model_name=cfg.VLLM_MODEL_NAME
            )
    
    if pipeline_steps['refinement'] or pipeline_steps['back_translation']:
        gemini_client = GeminiClient(
            api_keys=cfg.GEMINI_API_KEY,
            model_name=cfg.GEMINI_MODEL_NAME
        )

    source_table_paths = sorted(list(cfg.TABLES_DIR.glob("*.json")))
    print(f"\nFound {len(source_table_paths)} source tables to translate.")
    print(f"Target languages: {len(cfg.LANGUAGES)} ({', '.join(cfg.LANGUAGES.keys())})")
    print(f"Total translations: {len(source_table_paths) * len(cfg.LANGUAGES)}")
    print(f"vLLM Mode: {VLLM_MODE.upper()}")
    if getattr(cfg, 'VLLM_IS_THINKING_MODEL', False):
        print(f"Model Type: THINKING MODEL ({cfg.VLLM_MODEL_NAME})")
    else:
        print(f"Model: {cfg.VLLM_MODEL_NAME}")
    print()

    skipped_tables_count = 0
    processed_tables_count = 0

    for table_idx, table_path in enumerate(tqdm(source_table_paths, desc="Processing Tables"), 1):
        table_id = table_path.stem
        # Only process tables starting with arxiv, finqa, or wiki
        if (table_id.startswith("arxiv")):
            continue
        
        if check_translation_completion(table_id, cfg.LANGUAGES):
            cprint(f"Skipping table {table_id}: Already translated into all target languages.", "yellow")
            skipped_tables_count += 1
            continue

        try:
            print(f"\n{'─'*70}")
            print(f"Table {table_idx}/{len(source_table_paths)}: {table_id}")
            print(f"{'─'*70}")
            
            translator = TableTranslator(table_id, table_path, vllm_client, gemini_client)
            translator.run_batch_translation(cfg.LANGUAGES, pipeline_steps)
            processed_tables_count += 1
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process table {table_id}: {e}\n")
            import traceback
            traceback.print_exc()

    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "="*70)
    print("  PHASE 1.2: TRANSLATION PIPELINE COMPLETE")
    print(f"  Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    if processed_tables_count > 0:
        print(f"  Average time per *processed* table: {elapsed/processed_tables_count:.2f}s")
    else:
        print("  No new tables were processed.")
    print(f"  Tables skipped (already complete): {skipped_tables_count}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()