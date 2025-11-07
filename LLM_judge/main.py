# run_evaluation.py

import os
import json
import logging
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Model Configuration ---
# The final list of models to be tested on the L4 GPU.
SUPPORTED_MODELS = {
    "qwen3-30b-thinking": {
        "id": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "notes": "The #1 choice: S-Tier reasoning, 'Thinking' fine-tune, and native multilingual support."
    },
    "mixtral-8x7b": {
        "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "notes": "Powerful MoE model. A great comparison point for reasoning vs. multilingual capability."
    },
    "deepseek-qwen3-8b": {
        "id": "DeepSeek-R1/0528-Qwen3-8B",
        "notes": "A high-performing 8B model merge. Serves as a strong baseline."
    }
}

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are a meticulous and impartial AI Quality Analyst. Your task is to critically evaluate an AI model's answer against a ground-truth "golden answer." Your evaluation process will depend on the `question_type` provided.

## CONTEXT
- **Question ID (Table Reference):** {question_id}
- **Question Type:** {question_type}
- **Question:** {question}

## ANSWERS TO EVALUATE
- **Golden Answer (Ground Truth):** {golden_answer}
- **Model Answer:** {model_answer}

## EVALUATION PROCESS

Your first step is to identify the `question_type`. Then, follow the specific instructions for that type.

---
### **PATH A: IF Question Type is 'value' (or similar, e.g., anything not 'open_ended_reasoning')**
Your goal is to verify factual accuracy with a precise verdict.
**1. Analysis:** Compare the Model Answer directly against the Golden Answer. Determine if they are an exact match, semantically equivalent, partially correct, or incorrect.
**2. Verdict Definitions:**
   - **`exact_match`**: The model's answer is identical to the golden answer.
   - **`equivalent`**: The answer is semantically identical but has minor formatting/phrasing differences.
   - **`partially_correct`**: The answer contains some correct information but is incomplete or includes extra incorrect info.
   - **`incorrect`**: The answer is factually wrong.
**3. Scoring:** `exact_match`: 3, `equivalent`: 2, `partially_correct`: 1, `incorrect`: 0.

---
### **PATH B: IF Question Type is 'open_ended_reasoning'**
Your goal is to assess the overall quality of the explanation.
**1. Analysis (Chain of Thought):** Analyze Correctness, Completeness, and Clarity.
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
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')

class HuggingFaceEvaluator:
    """A class to handle loading and running a Hugging Face model on a GPU."""
    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"INFO: Using device: {self.device}")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"INFO: Loading model '{model_id}'. This may take a few minutes...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("INFO: Model loaded successfully.")

    def evaluate(self, prompt):
        """Generates a response from the loaded model and extracts the JSON."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                logging.error(f"No JSON object found in response: {response_text}")
                return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {e}. Response was: {response_text}")
            return None

def process_item(item, evaluator):
    """Formats the prompt for a single item and calls the evaluator."""
    try:
        # --- THIS IS THE ONLY LINE THAT CHANGED ---
        # It will now look for 'model_answer' OR 'model_response' in your data file.
        model_answer = item.get('model_answer', item.get('model_response'))

        prompt = PROMPT_TEMPLATE.format(
            question_id=item['question_id'],
            question=item['question'],
            # The prompt expects 'question_type', but your data doesn't have it.
            # We can use 'reasoning_category' instead.
            question_type=item.get('reasoning_category', 'not specified'),
            golden_answer=str(item['golden_answer']),
            model_answer=str(model_answer)
        )
        scores = evaluator.evaluate(prompt)
        if scores:
            return {**item, **scores}
        else:
            return {**item, 'error': 'Failed to get a valid evaluation response'}
    except KeyError as e:
        logging.error(f"Missing key in item {item.get('question_id', 'N/A')}: {e}")
        return {**item, 'error': f'Missing key: {e}'}

def load_data(file_path):
    """Loads data from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main(args):
    setup_logging()
    
    if args.model_key not in SUPPORTED_MODELS:
        print(f"ERROR: Invalid model key '{args.model_key}'.")
        print(f"Choose from: {list(SUPPORTED_MODELS.keys())}")
        return

    model_id = SUPPORTED_MODELS[args.model_key]['id']
    output_file = f"results_{args.model_key}.jsonl"

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found at '{args.input}'")
        return

    evaluator = HuggingFaceEvaluator(model_id)
    all_items = load_data(args.input)
    
    processed_ids = set()
    if os.path.exists(output_file) and not args.overwrite:
        print(f"WARNING: Output file '{output_file}' already exists. Resuming...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed_ids.add(json.loads(line)['question_id'])
                except (json.JSONDecodeError, KeyError): continue
        print(f"INFO: Found {len(processed_ids)} already processed items.")
        
    items_to_process = [item for item in all_items if item['question_id'] not in processed_ids]
    
    if not items_to_process:
        print("INFO: All items have already been processed for this model."); return

    print(f"INFO: Starting evaluation for {len(items_to_process)} items using model '{model_id}'.")
    print(f"INFO: Results will be saved to '{output_file}'.")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {executor.submit(process_item, item, evaluator): item for item in items_to_process}
            
            for future in tqdm(as_completed(future_to_item), total=len(items_to_process), desc=f"Evaluating with {args.model_key}"):
                result = future.result()
                if result:
                    f_out.write(json.dumps(result) + '\n')
                    f_out.flush()

    print(f"✅ Evaluation complete for model '{args.model_key}'! Results saved to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch LLM-as-a-Judge evaluation on Hugging Face models.")
    parser.add_argument('model_key', type=str, choices=SUPPORTED_MODELS.keys(), help="The key for the model to use for evaluation.")
    parser.add_argument('-i', '--input', type=str, help="Path to the input JSONL file.")
    parser.add_argument('-w', '--workers', type=int, default=2, help="Number of parallel data processing workers.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing output file instead of resuming.")
    
    args = parser.parse_args()
    main(args)