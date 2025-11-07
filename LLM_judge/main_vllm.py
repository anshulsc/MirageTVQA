import os
import json
import logging
import time
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- Model Configuration ---
SUPPORTED_MODELS = {
    "mixtral-8x7b": {
        "id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ",
        "quantization": "awq",
        "max_len": 4096,
        "recommended_batch": 128,
    },
    "deepseek-qwen3-8b": {
        "id": "hxac/DeepSeek-R1-0528-Qwen3-8B-AWQ-4bit",
        "quantization": None,
        "max_len": 8192,
        "recommended_batch": 1024,
    },
    "llama3-8b": {
        "id": "casperhansen/Llama-3-8B-Instruct-AWQ",
        "quantization": "awq",
        "max_len": 4096,
        "recommended_batch": 128,
    }
}

# --- Prompt Template (Updated with Table Context) ---
PROMPT_TEMPLATE = """You are a meticulous and impartial LLM Evaluator. Your task is to critically evaluate an AI model's answer against a ground-truth "golden answer." Your evaluation process will depend on the `question_type` provided.

## CONTEXT
- **Question Type:** {question_type}
- **Question:** {question}

## TABLE CONTEXT
The question is based on the following table data:
{table_data}

## ANSWERS TO EVALUATE
- **Golden Answer (Ground Truth):** {golden_answer}
- **Model Answer:** {model_answer}

## EVALUATION PROCESS

Your first step is to identify the `question_type`. Then, follow the specific instructions for that type.

---
### **PATH A: IF Question Type is 'value' (or similar, e.g., anything not 'open_ended_reasoning')**
Your goal is to verify factual accuracy with a precise verdict.
**1. Analysis:** Compare the Model Answer directly against the Golden Answer. Use the table context to verify facts. Determine if they are an exact match, semantically equivalent, partially correct, or incorrect.
**2. Verdict Definitions:**
   - **`exact_match`**: The model's answer is identical to the golden answer.
   - **`equivalent`**: The answer is semantically identical but has minor formatting/phrasing differences.
   - **`partially_correct`**: The answer contains some correct information but is incomplete or includes extra incorrect info.
   - **`incorrect`**: The answer is factually wrong.
**3. Scoring:** `exact_match`: 3, `equivalent`: 2, `partially_correct`: 1, `incorrect`: 0.

---
### **PATH B: IF Question Type is 'open_ended_reasoning'**
Your goal is to assess the overall quality of the explanation.
**1. Analysis (Chain of Thought):** Analyze Correctness, Completeness, and Clarity. Use the table context to verify factual claims.
**2. Scoring Rubric (1-10 for each):**
   - **Correctness (Weight: 50%):** Factual accuracy and logical soundness.
   - **Completeness (Weight: 30%):** Addresses all key points.
   - **Clarity (Weight: 20%):** Well-written and understandable.
**3. Overall Score Calculation:** Weighted average: `(Correctness * 0.5) + (Completeness * 0.3) + (Clarity * 0.2)`

---
## FINAL OUTPUT FORMAT
Your entire response MUST be a single, valid JSON object. Do not include any text outside the JSON structure.

**For PATH A ('value' type):**
{{
  "question_type_evaluated": "value",
  "reasoning": "<Brief justification for your verdict.>",
  "verdict": "<'exact_match', 'equivalent', 'partially_correct', or 'incorrect'>",
  "accuracy_score": <0, 1, 2, or 3>,
  "final_verdict": "<'accepted' if verdict is 'exact_match' or 'equivalent', else 'rejected'>",
  "confidence_score": <0-100>
}}

**For PATH B ('open_ended_reasoning' type):**
{{
  "question_type_evaluated": "open_ended_reasoning",
  "reasoning": "<Summary of your Correctness, Completeness, and Clarity analysis.>",
  "scores": {{ "correctness": <score_1_to_10>, "completeness": <score_1_to_10>, "clarity": <score_1_to_10> }},
  "overall_score": <weighted_average_score_rounded_to_1_decimal>,
  "final_verdict": "<'accepted' if overall_score is 7.0 or higher, else 'rejected'>",
  "confidence_score": <0-100>
}}
"""

def setup_logging(log_file='evaluation.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO, 
        filename=log_file, 
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_table_from_file(table_name, tables_dir):
    """
    Load table data from a JSON file in the tables directory.
    
    Args:
        table_name: Name of the table file (without extension)
        tables_dir: Directory containing table files
        
    Returns:
        Formatted table string or error message
    """
    table_path = os.path.join(tables_dir, f"{table_name}.json")
    
    if not os.path.exists(table_path):
        error_msg = f"Table file not found: {table_path}"
        logging.warning(error_msg)
        return error_msg
    
    try:
        with open(table_path, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
        return format_table_structure(table_data)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON from {table_path}: {e}"
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Failed to load table from {table_path}: {e}"
        logging.error(error_msg)
        return error_msg

def format_table_structure(table_data):
    """
    Format table data structure into a readable string.
    """
    if isinstance(table_data, str):
        return table_data
    elif isinstance(table_data, dict):
        return json.dumps(table_data, indent=2)
    elif isinstance(table_data, list):
        if len(table_data) > 0 and isinstance(table_data[0], dict):
            # List of dictionaries (rows)
            formatted = []
            headers = list(table_data[0].keys())
            formatted.append(" | ".join(headers))
            formatted.append("-" * (len(" | ".join(headers))))
            for row in table_data:
                formatted.append(" | ".join(str(row.get(h, '')) for h in headers))
            return "\n".join(formatted)
        else:
            return json.dumps(table_data, indent=2)
    else:
        return str(table_data)

def extract_table_name_from_question_id(question_id):
    """
    Extract table name from question_id.
    Example: "arxiv_0ce0508a37_01" -> "arxiv_0ce0508a37"
    
    Args:
        question_id: The question ID string
        
    Returns:
        Table name (question_id without the last part after final underscore)
    """
    parts = question_id.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    return question_id  # Return as-is if no underscore found

class vLLMEvaluator:
    """A class to handle loading and running a vLLM model for batch evaluation."""
    
    def __init__(self, model_id, quantization_method, max_model_len, 
                 tensor_parallel_size=1, gpu_memory_utilization=1.0,
                 enable_chunked_prefill=True, max_num_batched_tokens=None,
                 tables_dir='tables'):
        """
        Initialize the vLLM evaluator with optimizations for 24GB GPU.
        
        Args:
            model_id: HuggingFace model identifier
            quantization_method: Quantization method (e.g., 'awq', None)
            max_model_len: Maximum sequence length for the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            enable_chunked_prefill: Enable chunked prefill for better batching
            max_num_batched_tokens: Max tokens to batch together (controls memory)
            tables_dir: Directory containing table files
        """
        self.tables_dir = tables_dir
        print(f"INFO: Initializing vLLM for model '{model_id}'.")
        print(f"INFO: Tables directory: '{tables_dir}'")
        
        if quantization_method:
            print(f"INFO: Using quantization method: '{quantization_method}'")
        else:
            print("INFO: Loading model in its native precision (no quantization).")
        
        print(f"INFO: Setting max model length to: {max_model_len}")
        
        # Calculate optimal max_num_batched_tokens if not provided
        if max_num_batched_tokens is None:
            if quantization_method == "awq":
                max_num_batched_tokens = 8192
            else:
                max_num_batched_tokens = 4096
        
        print(f"INFO: Max batched tokens: {max_num_batched_tokens}")
        print(f"INFO: Chunked prefill: {enable_chunked_prefill}")
        
        # Initialize vLLM with optimizations
        self.llm = LLM(
            model=model_id,
            quantization=quantization_method,
            dtype='auto',
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=False,
            disable_log_stats=True,
        )
        
        # Optimized sampling params for evaluation (deterministic)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            top_p=1.0,
            skip_special_tokens=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("INFO: vLLM engine loaded successfully.")
        
    def evaluate_batch(self, batch_items):
        """
        Evaluate a batch of items using the vLLM model.
        Uses vLLM's continuous batching for optimal throughput.
        
        Args:
            batch_items: List of dictionaries containing question data
            
        Returns:
            List of evaluation results as dictionaries
        """
        prompts = []
        
        for item in batch_items:
            try:
                model_answer = item.get('model_answer', item.get('model_response', ''))
                question_id = item.get('question_id', '')
                
                # Extract table name from question_id and load table
                table_name = extract_table_name_from_question_id(question_id)
                table_str = load_table_from_file(table_name, self.tables_dir)
                
                # Create the prompt using the template
                messages = [{
                    "role": "user", 
                    "content": PROMPT_TEMPLATE.format(
                        question=item['question'],
                        question_type=item.get('reasoning_category', 
                                              item.get('question_type', 'not specified')),
                        table_data=table_str,
                        golden_answer=str(item['golden_answer']),
                        model_answer=str(model_answer)
                    )
                }]
                
                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                prompts.append(prompt)
                
            except KeyError as e:
                logging.error(f"Missing key in item {item.get('question_id', 'N/A')}: {e}")
                prompts.append("")

        # Generate responses - vLLM automatically uses continuous batching
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        
        # Parse results
        results = []
        for idx, output in enumerate(outputs):
            response_text = output.outputs[0].text
            
            try:
                # Extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response_text[json_start:json_end]
                    parsed_result = json.loads(json_str)
                    results.append(parsed_result)
                else:
                    logging.error(f"No JSON object found in vLLM response: {response_text}")
                    results.append({
                        'error': 'No JSON object found',
                        'raw_response': response_text
                    })
                    
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing failed: {e}. Response was: {response_text}")
                results.append({
                    'error': f'JSON parsing failed: {str(e)}',
                    'raw_response': response_text
                })
                
        return results

def load_data(file_path):
    """Load JSONL data from file."""
    print(f"INFO: Loading data from '{file_path}'")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"INFO: Loaded {len(data)} items")
    return data

def create_batches(data, batch_size):
    """Create batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main(args):
    """Main evaluation function."""
    setup_logging()
    
    # Validate model selection
    if args.model_key not in SUPPORTED_MODELS:
        print(f"ERROR: Invalid model key '{args.model_key}'.")
        print(f"Choose from: {list(SUPPORTED_MODELS.keys())}")
        return

    # Get model configuration
    model_info = SUPPORTED_MODELS[args.model_key]
    model_id = model_info['id']
    quant_method = model_info.get('quantization')
    
    # Use model-specific max_len if not explicitly provided
    if args.max_model_len is None:
        max_model_len = model_info.get('max_len', 4096)
        print(f"INFO: Using default max_model_len={max_model_len} for {args.model_key}")
    else:
        max_model_len = args.max_model_len
    
    # Use recommended batch size if not specified
    if args.batch_size is None:
        batch_size = model_info.get('recommended_batch', 64)
        print(f"INFO: Using recommended batch_size={batch_size} for {args.model_key}")
    else:
        batch_size = args.batch_size
    
    output_file = f"results_vllm_{args.model_key}.jsonl"

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found at '{args.input}'")
        return

    # Initialize evaluator with optimizations
    print(f"\n{'='*60}")
    print(f"Starting Optimized Batch Evaluation with vLLM")
    print(f"{'='*60}")
    print(f"Model: {args.model_key} ({model_id})")
    print(f"Input: {args.input}")
    print(f"Output: {output_file}")
    print(f"Batch size: {batch_size}")
    print(f"Max model length: {max_model_len}")
    print(f"GPU memory utilization: {args.gpu_mem}")
    print(f"Max batched tokens: {args.max_batched_tokens}")
    print(f"Tables directory: {args.tables_dir}")
    print(f"{'='*60}\n")
    
    # Verify tables directory exists
    if not os.path.exists(args.tables_dir):
        print(f"WARNING: Tables directory '{args.tables_dir}' does not exist!")
        print(f"Creating directory: {args.tables_dir}")
        os.makedirs(args.tables_dir, exist_ok=True)
    
    evaluator = vLLMEvaluator(
        model_id, 
        quantization_method=quant_method, 
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_batched_tokens,
        tables_dir=args.tables_dir
    )
    
    # Load all items
    all_items = load_data(args.input)
    
    # Check for existing results and resume if needed
    processed_ids = set()
    if os.path.exists(output_file) and not args.overwrite:
        print(f"WARNING: Output file '{output_file}' already exists.")
        print("INFO: Resuming from existing progress...")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['question_id'])
                except (json.JSONDecodeError, KeyError):
                    continue
                    
        print(f"INFO: Found {len(processed_ids)} already processed items.")
    
    # Filter out already processed items
    items_to_process = [
        item for item in all_items 
        if item['question_id'] not in processed_ids
    ]
    
    if not items_to_process:
        print("INFO: All items have already been processed for this model.")
        print("Use --overwrite flag to re-run evaluation.")
        return

    print(f"\nINFO: Starting evaluation for {len(items_to_process)} items.")
    print(f"INFO: Using vLLM with continuous batching for optimal throughput")
    print(f"INFO: Table context will be included in evaluation prompts\n")
    
    # Create batches
    item_batches = list(create_batches(items_to_process, batch_size))
    print(f"INFO: Processing {len(item_batches)} batches\n")

    # Process batches with progress bar
    start_time = time.time()
    total_processed = 0
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for batch_idx, batch in enumerate(tqdm(item_batches, desc=f"Evaluating with {args.model_key}")):
            try:
                batch_results = evaluator.evaluate_batch(batch)
                
                # Write results
                for original_item, result in zip(batch, batch_results):
                    final_record = {**original_item, **result}
                    f_out.write(json.dumps(final_record) + '\n')
                    f_out.flush()
                    total_processed += 1
                    
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                print(f"\nERROR: Failed to process batch {batch_idx}: {e}")
                continue

    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total items processed: {total_processed}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per item: {elapsed_time/total_processed:.2f} seconds")
    print(f"Throughput: {total_processed/elapsed_time:.2f} items/second")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optimized batch LLM-as-a-Judge evaluation using vLLM with continuous batching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'model_key', 
        type=str, 
        choices=SUPPORTED_MODELS.keys(), 
        help="The key for the model to use."
    )
    
    parser.add_argument(
        '-i', '--input', 
        default="/teamspace/studios/this_studio/LLM_judge/google_gemma-3-4b-it_multitableqa_clean_default_20251022_042714.jsonl",
        type=str, 
        help="Path to the input JSONL file containing questions and answers."
    )
    
    parser.add_argument(
        '--tables-dir',
        type=str,
        default='./tables',
        help="Directory containing table JSON files (e.g., 'arxiv_0ce0508a37.json')"
    )
    
    parser.add_argument(
        '-b', '--batch-size', 
        type=int, 
        default=1024,
        help="Number of prompts to process in a batch. If not set, uses model-specific recommended value."
    )
    
    parser.add_argument(
        '--gpu-mem', 
        type=float, 
        default=0.90, 
        help="GPU memory utilization for vLLM (e.g., 0.9 for 90%%)."
    )
    
    parser.add_argument(
        '--max-model-len', 
        type=int, 
        default=None,
        help="Maximum sequence length for the model. If not set, uses model-specific default."
    )
    
    parser.add_argument(
        '--max-batched-tokens',
        type=int,
        default=16978,
        help="Max tokens to batch together. Controls memory usage. Auto-calculated if not set."
    )
    
    parser.add_argument(
        '--enable-chunked-prefill',
        action='store_true',
        default=True,
        help="Enable chunked prefill for better batching efficiency (recommended)."
    )
    
    parser.add_argument(
        '--overwrite', 
        action='store_true', 
        help="Overwrite existing output file instead of resuming."
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nINFO: Evaluation interrupted by user. Progress has been saved.")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFATAL ERROR: {e}")
        print("Check evaluation.log for details.")








# import os
# import json
# import logging
# import time
# import argparse
# from tqdm import tqdm
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer

# # --- Model Configuration ---
# SUPPORTED_MODELS = {
#     "mixtral-8x7b": {
#         "id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ",
#         "quantization": "awq",
#         "max_len": 4096,
#         "recommended_batch": 128,  # AWQ quantized, can handle larger batches
#     },
#     "deepseek-qwen3-8b": {
#         "id": "hxac/DeepSeek-R1-0528-Qwen3-8B-AWQ-4bit",
#         "quantization": None,
#         "max_len": 2096,
#         "recommended_batch": 1024,  # No quantization, moderate batch size
#     },
#     "llama3-8b": {
#         "id": "casperhansen/Llama-3-8B-Instruct-AWQ",
#         "quantization": "awq",
#         "max_len": 4096,
#         "recommended_batch": 128,  # AWQ quantized, can handle larger batches
#     }
# }

# # --- Prompt Template ---
# PROMPT_TEMPLATE = """You are a meticulous and impartial LLM Evaulator. Your task is to critically evaluate an AI model's answer against a ground-truth "golden answer." Your evaluation process will depend on the `question_type` provided.

# ## CONTEXT
# - **Question Type:** {question_type}
# - **Question:** {question}

# ## ANSWERS TO EVALUATE
# - **Golden Answer (Ground Truth):** {golden_answer}
# - **Model Answer:** {model_answer}

# ## EVALUATION PROCESS

# Your first step is to identify the `question_type`. Then, follow the specific instructions for that type.

# ---
# ### **PATH A: IF Question Type is 'value' (or similar, e.g., anything not 'open_ended_reasoning')**
# Your goal is to verify factual accuracy with a precise verdict.
# **1. Analysis:** Compare the Model Answer directly against the Golden Answer. Determine if they are an exact match, semantically equivalent, partially correct, or incorrect.
# **2. Verdict Definitions:**
#    - **`exact_match`**: The model's answer is identical to the golden answer.
#    - **`equivalent`**: The answer is semantically identical but has minor formatting/phrasing differences.
#    - **`partially_correct`**: The answer contains some correct information but is incomplete or includes extra incorrect info.
#    - **`incorrect`**: The answer is factually wrong.
# **3. Scoring:** `exact_match`: 3, `equivalent`: 2, `partially_correct`: 1, `incorrect`: 0.

# ---
# ### **PATH B: IF Question Type is 'open_ended_reasoning'**
# Your goal is to assess the overall quality of the explanation.
# **1. Analysis (Chain of Thought):** Analyze Correctness, Completeness, and Clarity.
# **2. Scoring Rubric (1-10 for each):**
#    - **Correctness (Weight: 50%):** Factual accuracy and logical soundness.
#    - **Completeness (Weight: 30%):** Addresses all key points.
#    - **Clarity (Weight: 20%):** Well-written and understandable.
# **3. Overall Score Calculation:** Weighted average: `(Correctness * 0.5) + (Completeness * 0.3) + (Clarity * 0.2)`

# ---
# ## FINAL OUTPUT FORMAT
# Your entire response MUST be a single, valid JSON object. Do not include any text outside the JSON structure.

# **For PATH A ('value' type):**
# {{
#   "question_type_evaluated": "value",
#   "reasoning": "<Brief justification for your verdict.>",
#   "verdict": "<'exact_match', 'equivalent', 'partially_correct', or 'incorrect'>",
#   "accuracy_score": <0, 1, 2, or 3>,
#   "final_verdict": "<'accepted' if verdict is 'exact_match' or 'equivalent', else 'rejected'>",
#   "confidence_score": <0-100>
# }}

# **For PATH B ('open_ended_reasoning' type):**
# {{
#   "question_type_evaluated": "open_ended_reasoning",
#   "reasoning": "<Summary of your Correctness, Completeness, and Clarity analysis.>",
#   "scores": {{ "correctness": <score_1_to_10>, "completeness": <score_1_to_10>, "clarity": <score_1_to_10> }},
#   "overall_score": <weighted_average_score_rounded_to_1_decimal>,
#   "final_verdict": "<'accepted' if overall_score is 7.0 or higher, else 'rejected'>",
#   "confidence_score": <0-100>
# }}
# """

# def setup_logging(log_file='evaluation.log'):
#     """Setup logging configuration."""
#     logging.basicConfig(
#         level=logging.INFO, 
#         filename=log_file, 
#         filemode='a',
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )

# class vLLMEvaluator:
#     """A class to handle loading and running a vLLM model for batch evaluation."""
    
#     def __init__(self, model_id, quantization_method, max_model_len, 
#                  tensor_parallel_size=1, gpu_memory_utilization=1.0,
#                  enable_chunked_prefill=True, max_num_batched_tokens=None):
#         """
#         Initialize the vLLM evaluator with optimizations for 24GB GPU.
        
#         Args:
#             model_id: HuggingFace model identifier
#             quantization_method: Quantization method (e.g., 'awq', None)
#             max_model_len: Maximum sequence length for the model
#             tensor_parallel_size: Number of GPUs for tensor parallelism
#             gpu_memory_utilization: Fraction of GPU memory to use
#             enable_chunked_prefill: Enable chunked prefill for better batching
#             max_num_batched_tokens: Max tokens to batch together (controls memory)
#         """
#         print(f"INFO: Initializing vLLM for model '{model_id}'.")
        
#         if quantization_method:
#             print(f"INFO: Using quantization method: '{quantization_method}'")
#         else:
#             print("INFO: Loading model in its native precision (no quantization).")
        
#         print(f"INFO: Setting max model length to: {max_model_len}")
        
#         # Calculate optimal max_num_batched_tokens if not provided
#         if max_num_batched_tokens is None:
#             # For 24GB GPU: Conservative estimate
#             if quantization_method == "awq":
#                 max_num_batched_tokens = 8192  # AWQ uses less memory
#             else:
#                 max_num_batched_tokens = 4096  # Non-quantized needs more memory
        
#         print(f"INFO: Max batched tokens: {max_num_batched_tokens}")
#         print(f"INFO: Chunked prefill: {enable_chunked_prefill}")
        
#         # Initialize vLLM with optimizations
#         self.llm = LLM(
#             model=model_id,
#             quantization=quantization_method,
#             dtype='auto',
#             tensor_parallel_size=tensor_parallel_size,
#             gpu_memory_utilization=gpu_memory_utilization,
#             max_model_len=max_model_len,
#             enable_chunked_prefill=enable_chunked_prefill,
#             max_num_batched_tokens=max_num_batched_tokens,
#             # Additional optimizations
#             enforce_eager=False,  # Use CUDA graphs for better performance
#             disable_log_stats=True,  # Reduce overhead
#         )
        
#         # Optimized sampling params for evaluation (deterministic)
#         self.sampling_params = SamplingParams(
#             temperature=0.0,  # Deterministic for evaluation
#             max_tokens=1024,
#             top_p=1.0,
#             skip_special_tokens=True,  # Clean output
#         )
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         print("INFO: vLLM engine loaded successfully.")
        
#     def evaluate_batch(self, batch_items):
#         """
#         Evaluate a batch of items using the vLLM model.
#         Uses vLLM's continuous batching for optimal throughput.
        
#         Args:
#             batch_items: List of dictionaries containing question data
            
#         Returns:
#             List of evaluation results as dictionaries
#         """
#         prompts = []
        
#         for item in batch_items:
#             try:
#                 model_answer = item.get('model_answer', item.get('model_response', ''))
                
#                 # Create the prompt using the template
#                 messages = [{
#                     "role": "user", 
#                     "content": PROMPT_TEMPLATE.format(
#                         question=item['question'],
#                         question_type=item.get('reasoning_category', 
#                                               item.get('question_type', 'not specified')),
#                         golden_answer=str(item['golden_answer']),
#                         model_answer=str(model_answer)
#                     )
#                 }]
                
#                 # Apply chat template
#                 prompt = self.tokenizer.apply_chat_template(
#                     messages, 
#                     tokenize=False, 
#                     add_generation_prompt=True
#                 )
#                 prompts.append(prompt)
                
#             except KeyError as e:
#                 logging.error(f"Missing key in item {item.get('question_id', 'N/A')}: {e}")
#                 prompts.append("")

#         # Generate responses - vLLM automatically uses continuous batching
#         outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        
#         # Parse results
#         results = []
#         for idx, output in enumerate(outputs):
#             response_text = output.outputs[0].text
            
#             try:
#                 # Extract JSON from response
#                 json_start = response_text.find('{')
#                 json_end = response_text.rfind('}') + 1
                
#                 if json_start != -1 and json_end != -1:
#                     json_str = response_text[json_start:json_end]
#                     parsed_result = json.loads(json_str)
#                     results.append(parsed_result)
#                 else:
#                     logging.error(f"No JSON object found in vLLM response: {response_text}")
#                     results.append({
#                         'error': 'No JSON object found',
#                         'raw_response': response_text
#                     })
                    
#             except json.JSONDecodeError as e:
#                 logging.error(f"JSON parsing failed: {e}. Response was: {response_text}")
#                 results.append({
#                     'error': f'JSON parsing failed: {str(e)}',
#                     'raw_response': response_text
#                 })
                
#         return results

# def load_data(file_path):
#     """Load JSONL data from file."""
#     print(f"INFO: Loading data from '{file_path}'")
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f]
#     print(f"INFO: Loaded {len(data)} items")
#     return data

# def create_batches(data, batch_size):
#     """Create batches from data."""
#     for i in range(0, len(data), batch_size):
#         yield data[i:i + batch_size]

# def main(args):
#     """Main evaluation function."""
#     setup_logging()
    
#     # Validate model selection
#     if args.model_key not in SUPPORTED_MODELS:
#         print(f"ERROR: Invalid model key '{args.model_key}'.")
#         print(f"Choose from: {list(SUPPORTED_MODELS.keys())}")
#         return

#     # Get model configuration
#     model_info = SUPPORTED_MODELS[args.model_key]
#     model_id = model_info['id']
#     quant_method = model_info.get('quantization')
    
#     # Use model-specific max_len if not explicitly provided
#     if args.max_model_len is None:
#         max_model_len = model_info.get('max_len', 4096)
#         print(f"INFO: Using default max_model_len={max_model_len} for {args.model_key}")
#     else:
#         max_model_len = args.max_model_len
    
#     # Use recommended batch size if not specified
#     if args.batch_size is None:
#         batch_size = model_info.get('recommended_batch', 64)
#         print(f"INFO: Using recommended batch_size={batch_size} for {args.model_key}")
#     else:
#         batch_size = args.batch_size
    
#     output_file = f"results_vllm_{args.model_key}.jsonl"

#     # Check input file exists
#     if not os.path.exists(args.input):
#         print(f"ERROR: Input file not found at '{args.input}'")
#         return

#     # Initialize evaluator with optimizations
#     print(f"\n{'='*60}")
#     print(f"Starting Optimized Batch Evaluation with vLLM")
#     print(f"{'='*60}")
#     print(f"Model: {args.model_key} ({model_id})")
#     print(f"Input: {args.input}")
#     print(f"Output: {output_file}")
#     print(f"Batch size: {batch_size}")
#     print(f"Max model length: {max_model_len}")
#     print(f"GPU memory utilization: {args.gpu_mem}")
#     print(f"Max batched tokens: {args.max_batched_tokens}")
#     print(f"{'='*60}\n")
    
#     evaluator = vLLMEvaluator(
#         model_id, 
#         quantization_method=quant_method, 
#         max_model_len=max_model_len,
#         gpu_memory_utilization=args.gpu_mem,
#         enable_chunked_prefill=args.enable_chunked_prefill,
#         max_num_batched_tokens=args.max_batched_tokens
#     )
    
#     # Load all items
#     all_items = load_data(args.input)
    
#     # Check for existing results and resume if needed
#     processed_ids = set()
#     if os.path.exists(output_file) and not args.overwrite:
#         print(f"WARNING: Output file '{output_file}' already exists.")
#         print("INFO: Resuming from existing progress...")
        
#         with open(output_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     processed_ids.add(json.loads(line)['question_id'])
#                 except (json.JSONDecodeError, KeyError):
#                     continue
                    
#         print(f"INFO: Found {len(processed_ids)} already processed items.")
    
#     # Filter out already processed items
#     items_to_process = [
#         item for item in all_items 
#         if item['question_id'] not in processed_ids
#     ]
    
#     if not items_to_process:
#         print("INFO: All items have already been processed for this model.")
#         print("Use --overwrite flag to re-run evaluation.")
#         return

#     print(f"\nINFO: Starting evaluation for {len(items_to_process)} items.")
#     print(f"INFO: Using vLLM with continuous batching for optimal throughput\n")
    
#     # Create batches
#     item_batches = list(create_batches(items_to_process, batch_size))
#     print(f"INFO: Processing {len(item_batches)} batches\n")

#     # Process batches with progress bar
#     start_time = time.time()
#     total_processed = 0
    
#     with open(output_file, 'a', encoding='utf-8') as f_out:
#         for batch_idx, batch in enumerate(tqdm(item_batches, desc=f"Evaluating with {args.model_key}")):
#             try:
#                 batch_results = evaluator.evaluate_batch(batch)
                
#                 # Write results
#                 for original_item, result in zip(batch, batch_results):
#                     final_record = {**original_item, **result}
#                     f_out.write(json.dumps(final_record) + '\n')
#                     f_out.flush()
#                     total_processed += 1
                    
#             except Exception as e:
#                 logging.error(f"Error processing batch {batch_idx}: {e}")
#                 print(f"\nERROR: Failed to process batch {batch_idx}: {e}")
#                 continue

#     elapsed_time = time.time() - start_time
    
#     print(f"\n{'='*60}")
#     print(f"✅ Evaluation Complete!")
#     print(f"{'='*60}")
#     print(f"Results saved to: {output_file}")
#     print(f"Total items processed: {total_processed}")
#     print(f"Time elapsed: {elapsed_time:.2f} seconds")
#     print(f"Average time per item: {elapsed_time/total_processed:.2f} seconds")
#     print(f"Throughput: {total_processed/elapsed_time:.2f} items/second")
#     print(f"{'='*60}\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run optimized batch LLM-as-a-Judge evaluation using vLLM with continuous batching.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
    
#     parser.add_argument(
#         'model_key', 
#         type=str, 
#         choices=SUPPORTED_MODELS.keys(), 
#         help="The key for the model to use."
#     )
    
#     parser.add_argument(
#         '-i', '--input', 
#         default="/teamspace/studios/this_studio/LLM_judge/google_gemma-3-4b-it_multitableqa_clean_default_20251022_042714.jsonl",
#         type=str, 
#         help="Path to the input JSONL file containing questions and answers."
#     )
    
#     parser.add_argument(
#         '-b', '--batch-size', 
#         type=int, 
#         default=1024,
#         help="Number of prompts to process in a batch. If not set, uses model-specific recommended value."
#     )
    
#     parser.add_argument(
#         '--gpu-mem', 
#         type=float, 
#         default=0.90, 
#         help="GPU memory utilization for vLLM (e.g., 0.9 for 90%%)."
#     )
    
#     parser.add_argument(
#         '--max-model-len', 
#         type=int, 
#         default=None,
#         help="Maximum sequence length for the model. If not set, uses model-specific default."
#     )
    
#     parser.add_argument(
#         '--max-batched-tokens',
#         type=int,
#         default=16978,
#         help="Max tokens to batch together. Controls memory usage. Auto-calculated if not set."
#     )
    
#     parser.add_argument(
#         '--enable-chunked-prefill',
#         action='store_true',
#         default=True,
#         help="Enable chunked prefill for better batching efficiency (recommended)."
#     )
    
#     parser.add_argument(
#         '--overwrite', 
#         action='store_true', 
#         help="Overwrite existing output file instead of resuming."
#     )
    
#     args = parser.parse_args()
    
#     try:
#         main(args)
#     except KeyboardInterrupt:
#         print("\n\nINFO: Evaluation interrupted by user. Progress has been saved.")
#     except Exception as e:
#         logging.error(f"Fatal error: {e}", exc_info=True)
#         print(f"\nFATAL ERROR: {e}")
#         print("Check evaluation.log for details.")