import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from termcolor import cprint
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel, Field
from collections import defaultdict
import threading
from queue import Queue

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'


class GoldenAnswerVerification(BaseModel):
    """Pydantic model for structured golden answer verification output."""
    reasoning: str = Field(description="Detailed analysis comparing golden answer with model responses and table evidence")
    is_golden_correct: str = Field(description="Either 'Correct', 'Incorrect', or 'Needs_Revision'")
    corrected_answer: List[str] = Field(description="The correct answer(s) if golden answer needs revision, otherwise same as golden answer")
    confidence: str = Field(description="Confidence level: 'High', 'Medium', or 'Low'")
    evidence_summary: str = Field(description="Brief summary of key evidence from table supporting the decision")


class VLLMGoldenAnswerJudge:
    """vLLM-based judge for verifying golden answers against multiple model responses."""
    
    def __init__(
        self,
        model_name: str = "/home/anshulsc/links/scratch/cache/hub/models--Qwen--Qwen3-Next-80B-A3B-Thinking/snapshots/e502dd4100cc68c0de57643fd4317ec93a128670",
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 32768,
        enable_thinking: bool = True,
        use_guided_decoding: bool = True
    ):
        """Initialize the vLLM Golden Answer Judge."""
        cprint(f"Initializing vLLM Golden Answer Verification Judge", "blue")
        cprint(f"  Model: {model_name}", "yellow")
        cprint(f"  Tensor Parallel Size: {tensor_parallel_size}", "yellow")
        cprint(f"  GPU Memory Utilization: {gpu_memory_utilization}", "yellow")
        cprint(f"  Max Model Length: {max_model_len}", "yellow")
        cprint(f"  Thinking Mode: {enable_thinking}", "yellow")
        cprint(f"  Guided Decoding: {use_guided_decoding}", "yellow")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.use_guided_decoding = use_guided_decoding
        
        # Setup sampling parameters
        if use_guided_decoding:
            json_schema = GoldenAnswerVerification.model_json_schema()
            self.sampling_params = SamplingParams(
                temperature=0.6 if enable_thinking else 0.7,
                top_p=0.95 if enable_thinking else 0.9,
                top_k=20 if enable_thinking else 50,
                max_tokens=4096,
                stop=["<|im_end|>", "<|endoftext|>"],
                guided_decoding=GuidedDecodingParams(json=json_schema)
            )
        else:
            self.sampling_params = SamplingParams(
                temperature=0.6 if enable_thinking else 0.7,
                top_p=0.95 if enable_thinking else 0.9,
                top_k=20 if enable_thinking else 50,
                max_tokens=8192,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
        
        # Thread-safe lock for file writing
        self.write_lock = threading.Lock()
        
        cprint(f"✓ vLLM Golden Answer Judge initialized successfully!", "green")
    
    def json_table_to_markdown(self, table_data: Dict) -> str:
        """Convert JSON table format to markdown table."""
        if not table_data or "data" not in table_data:
            return "No table data available"
        
        data = table_data["data"]
        if not data or len(data) == 0:
            return "Empty table"
        
        # First row is typically the header
        headers = data[0]
        rows = data[1:]
        
        # Create markdown table
        markdown_lines = []
        
        # Header row
        markdown_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        
        # Separator row
        markdown_lines.append("| " + " | ".join("---" for _ in headers) + " |")
        
        # Data rows
        for row in rows:
            markdown_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(markdown_lines)
    
    def create_verification_prompt(
        self, 
        question_id: str,
        question: str,
        question_type: str,
        golden_answer: List[str],
        model_responses: List[Dict[str, Any]],
        table_data: Dict
    ) -> str:
        """Create a prompt for verifying the golden answer."""
        
        
        thinking_instruction = ""
        if self.enable_thinking:
            thinking_instruction = """
IMPORTANT: Use your thinking mode to reason through this verification carefully. 
Think step-by-step through your analysis before providing the final verdict.
"""
        
        # Format model responses
        model_responses_text = "\n\n".join([
            f"**Model {i+1} ({resp.get('model_name', 'Unknown')}):**\n{resp.get('model_response', 'N/A')}"
            for i, resp in enumerate(model_responses)
        ])
        prompt = f"""
You are an expert evaluator tasked with verifying the correctness of golden (ground truth) answers for table-based questions. Your role is critical: you must determine whether synthetically generated QA pairs have the correct golden answer by analyzing the table data and comparing it with responses from multiple AI models.

Optional: {thinking_instruction}

YOUR MISSION
You are given:
1. A table containing factual data
2. A question about the table
3. The question type (e.g., Numerical Aggregation, Comparative Reasoning, etc.)
4. The GOLDEN ANSWER (the supposed ground truth)
5. Responses from 3 different AI models attempting to answer the same question

Your job is to:
- Verify if the golden answer is factually correct based on the table data
- Cross-reference with the model responses to identify consensus or discrepancies
- Determine if the golden answer needs revision
- Provide the corrected answer if needed

CRITICAL VERIFICATION RULES

**Evidence from Table:**
- ALWAYS verify claims directly against the table data
- Reference specific cells, rows, columns with coordinates (e.g., "Row 3, Column 'Sales': 1,200")
- For numeric questions, show step-by-step calculations
- For comparative questions, explicitly compare relevant values
- For list questions, ensure completeness (no missing or extra items)

**Model Response Analysis:**
- If 2+ models agree and their answer matches table data → Strong signal the golden answer may be wrong if it differs
- If all 3 models disagree with golden answer → High likelihood golden answer is incorrect
- If models agree with golden answer → Likely correct, but still verify against table
- If models all give different answers → Golden answer may be ambiguous or models are wrong; defer to table data

**Answer Format Precision:**
- Golden answer format must match question requirements exactly
- Single value questions should have single values, not explanations
- List questions should have complete lists in proper order if specified
- Numeric answers should match precision and units requested
- No extra explanatory text unless question explicitly asks for it

**Verdict Guidelines:**
- **Correct**: Golden answer is factually accurate and format is appropriate
- **Incorrect**: Golden answer is factually wrong based on table evidence
- **Needs_Revision**: Golden answer is partially correct but needs format adjustment, completion, or minor correction

**Confidence Levels:**
- **High**: Clear table evidence + model consensus
- **Medium**: Table evidence clear but models mixed OR table evidence requires interpretation
- **Low**: Ambiguous table data or question interpretation unclear

OUTPUT FORMAT (Required - JSON only)
Produce only a single JSON object with these exact fields:

{{
  "reasoning": "Detailed step-by-step analysis. Include: (1) What the golden answer states, (2) What the table data shows with specific cell references, (3) What the models answered and their consensus level, (4) Your verification logic and calculations if needed, (5) Why you reached your verdict",
  
  "is_golden_correct": "Correct" | "Incorrect" | "Needs_Revision",
  
  "corrected_answer": ["The correct answer(s) in the exact format required. If golden is correct, repeat it here. If incorrect, provide the factually correct answer based on table data."],
  
  "confidence": "High" | "Medium" | "Low",
  
  "evidence_summary": "Brief summary (2-3 sentences) of key table evidence that supports your decision"
}}

EXAMPLE (for guidance — follow this style)
Given this table:

| Asset Type         | Description   | Quantity / Value |
| ------------------ | ------------- | ---------------- |
| U.S. Treasury Bond | 5-year        | $10,000          |
| U.S. Treasury Bond | 10-year       | $10,000          |
| Futures Contracts  | Gold          | 600 ounces       |
| Futures Contracts  | Crude Oil     | 2,000 barrels    |
| Equity             | Exxon Mobil   | 10,000 shares    |
| Equity             | AT&T          | 10,000 shares    |
| Currency Pair      | Long EUR/USD  | 100,000 EUR      |
| Currency Pair      | Short USD/JPY | 10,000,000 JPY   |

Question: "Which asset has the longest maturity period, based on its description?"
Provided Answer: `[["10-year"]]`
Evaluation (example JSON output you should emulate):

{{
"reasoning": "The question asks for the asset (i.e., the asset type or row) with the longest maturity period. The provided answer '10-year' is only the description value, not the asset type. The table shows two 'U.S. Treasury Bond' rows with descriptions '5-year' and '10-year' (rows 1 and 2). The correct asset (by asset type) that contains the '10-year' description is 'U.S. Treasury Bond'. Therefore the provided answer is incomplete and mis-formatted: it returns the description rather than the asset. Evidence: table rows with 'U.S. Treasury Bond' and '10-year'.",
 "is_golden_correct": "Incorrect" ,
  
  "corrected_answer": ["U.S. Treasuary"],
  
  "confidence": "High",
  
  "evidence_summary": "Brief summary (2-3 sentences) of key table evidence that supports your decision"
}}

ADDITIONAL GUIDANCE

EXAMPLE VERIFICATION SCENARIOS

**Scenario 1: Golden answer is correct**
- Golden: ["42"]
- Models: ["42", "42", "41"]
- Table shows: Sum of column = 42
→ Verdict: "Correct" (2/3 models agree + table confirms)

**Scenario 2: Golden answer is incorrect**
- Golden: ["Company A"]
- Models: ["Company B", "Company B", "Company B"]
- Table shows: Row with highest sales is Company B with $1.5M vs Company A with $1.2M
→ Verdict: "Incorrect", corrected_answer: ["Company B"]

**Scenario 3: Golden answer needs revision**
- Golden: ["The answer is 15% based on..."] (explanation included)
- Models: ["15%", "15", "15%"]
- Question asks: "What percentage..."
- Table shows: 15%
→ Verdict: "Needs_Revision", corrected_answer: ["15%"] (remove explanation)

NOW VERIFY THE FOLLOWING:

**QUESTION ID:** {question_id}

**TABLE DATA:**
{json.dumps(table_data, indent=2)}

**QUESTION:**
{question}

**QUESTION TYPE:**
{question_type}

**GOLDEN ANSWER (To be verified):**
{json.dumps(golden_answer)}

**MODEL RESPONSES:**
{model_responses_text}

VERIFICATION INSTRUCTIONS:
1. First, independently verify what the correct answer should be based ONLY on the table data
2. Then compare your finding with the golden answer
3. Consider the model responses as additional signals (especially if they agree)
4. Make your final determination based on table evidence as the ultimate source of truth
5. Provide step-by-step reasoning that another expert could audit

Remember: Your primary duty is to the factual accuracy of the golden answer. Be rigorous and uncompromising about correctness.
"""
        
        
        return prompt
    
    def generate_verification_batch(
        self,
        batch: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> List[Dict]:
        """Generate verifications for a batch of QA pairs."""
        results = [None] * len(batch)
        
        for attempt in range(max_retries):
            tokenizer = self.llm.get_tokenizer()
            
            remaining_prompts = []
            remaining_indices = []
            
            for idx, item in enumerate(batch):
                if results[idx] is None:
                    prompt = self.create_verification_prompt(
                        question_id=item["question_id"],
                        question=item["question"],
                        question_type=item["question_type"],
                        golden_answer=item["golden_answer"],
                        model_responses=item["model_responses"],
                        table_data=item["table_data"]
                    )
                    messages = [{"role": "user", "content": prompt}]
                    
                    templated_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    )
                    
                    remaining_prompts.append(templated_text)
                    remaining_indices.append(idx)
            
            if not remaining_prompts:
                break
            
            cprint(f"\nBatch verification: {len(remaining_prompts)} prompts (attempt {attempt + 1}/{max_retries})...", "cyan")
            
            try:
                outputs = self.llm.generate(remaining_prompts, self.sampling_params)
                
                for i, output in enumerate(outputs):
                    original_idx = remaining_indices[i]
                    item = batch[original_idx]
                    generated_text = output.outputs[0].text
                    
                    try:
                        # Handle thinking tags if present
                        if "<think>" in generated_text and "</think>" in generated_text:
                            parts = generated_text.split("</think>")
                            if len(parts) > 1:
                                generated_text = parts[1].strip()
                        
                        # Remove markdown code blocks if present
                        if generated_text.startswith("```json"):
                            generated_text = generated_text[7:]
                        if generated_text.startswith("```"):
                            generated_text = generated_text[3:]
                        if generated_text.endswith("```"):
                            generated_text = generated_text[:-3]
                        generated_text = generated_text.strip()
                        
                        # Parse JSON
                        if self.use_guided_decoding:
                            verification_result = GoldenAnswerVerification.model_validate_json(generated_text)
                            result_dict = {
                                "question_id": item["question_id"],
                                "question": item["question"],
                                "question_type": item["question_type"],
                                "golden_answer": item["golden_answer"],
                                "model_responses": item["model_responses"],
                                "reasoning": verification_result.reasoning,
                                "is_golden_correct": verification_result.is_golden_correct,
                                "corrected_answer": verification_result.corrected_answer,
                                "confidence": verification_result.confidence,
                                "evidence_summary": verification_result.evidence_summary
                            }
                        else:
                            parsed_json = json.loads(generated_text)
                            result_dict = {
                                "question_id": item["question_id"],
                                "question": item["question"],
                                "question_type": item["question_type"],
                                "golden_answer": item["golden_answer"],
                                "model_responses": item["model_responses"],
                                "reasoning": parsed_json.get("reasoning", ""),
                                "is_golden_correct": parsed_json.get("is_golden_correct", "Unknown"),
                                "corrected_answer": parsed_json.get("corrected_answer", []),
                                "confidence": parsed_json.get("confidence", "Low"),
                                "evidence_summary": parsed_json.get("evidence_summary", "")
                            }
                        
                        results[original_idx] = result_dict
                        cprint(f"  ✓ {item['question_id']}: {result_dict['is_golden_correct']} ({result_dict['confidence']} confidence)", "green")
                        
                    except Exception as e:
                        cprint(f"  ✗ {item['question_id']}: Parse failed - {str(e)[:100]}", "yellow")
                        if attempt == max_retries - 1:
                            results[original_idx] = {
                                "question_id": item["question_id"],
                                "question": item["question"],
                                "question_type": item["question_type"],
                                "golden_answer": item["golden_answer"],
                                "model_responses": item["model_responses"],
                                "reasoning": f"Error parsing response: {str(e)}",
                                "is_golden_correct": "Error",
                                "corrected_answer": [],
                                "confidence": "Low",
                                "evidence_summary": "Verification failed"
                            }
                        
            except Exception as e:
                cprint(f"Batch verification failed: {e}", "red")
                if attempt == max_retries - 1:
                    for idx in remaining_indices:
                        if results[idx] is None:
                            item = batch[idx]
                            results[idx] = {
                                "question_id": item["question_id"],
                                "question": item["question"],
                                "question_type": item["question_type"],
                                "golden_answer": item["golden_answer"],
                                "model_responses": item["model_responses"],
                                "reasoning": f"Batch verification error: {str(e)}",
                                "is_golden_correct": "Error",
                                "corrected_answer": [],
                                "confidence": "Low",
                                "evidence_summary": "Verification failed"
                            }
        
        successful = sum(1 for r in results if r is not None and r["is_golden_correct"] != "Error")
        cprint(f"\n✓ Batch complete: {successful}/{len(batch)} successful", 
               "green" if successful == len(batch) else "yellow")
        
        return results
    
    def save_result_to_jsonl(self, result: Dict, output_file: str):
        """Thread-safe method to append a single result to the JSONL file."""
        with self.write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
    
    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, 'llm'):
            del self.llm
            cprint("vLLM engine cleaned up", "yellow")


def load_model_responses(jsonl_files: List[str]) -> Dict[str, List[Dict]]:
    """Load model responses from multiple JSONL files and organize by question_id."""
    responses_by_question = defaultdict(list)
    
    for jsonl_file in jsonl_files:
        cprint(f"Loading responses from: {jsonl_file}", "white")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    question_id = data.get('question_id')
                    if question_id:
                        responses_by_question[question_id].append({
                            'model_name': data.get('model_name', 'Unknown'),
                            'model_response': data.get('model_response', 'N/A')
                        })
                except Exception as e:
                    cprint(f"Error loading line: {e}", "yellow")
                    continue
    
    return responses_by_question


def load_qa_pairs_with_responses(
    qa_file: str,
    model_responses: Dict[str, List[Dict]]
) -> List[Dict]:
    """Load QA pairs and match with model responses."""
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    matched_data = []
    for qa_item in qa_data:
        question_id = qa_item['question_id']
        if question_id in model_responses:
            matched_data.append({
                **qa_item,
                'model_responses': model_responses[question_id]
            })
        else:
            cprint(f"Warning: No model responses found for {question_id}", "yellow")
    
    return matched_data


def load_table_data(table_file_path: str) -> Dict:
    """Load table data from JSON file."""
    with open(table_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_processed_questions(output_file: str) -> set:
    """Load already processed question IDs from the output file."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    processed.add(result['question_id'])
                except:
                    continue
    return processed


def prepare_file_batches(
    qa_files: List[Path],
    tables_dir: Path,
    model_responses: Dict[str, List[Dict]],
    processed_questions: set,
    batch_size: int
) -> List[Dict[str, Any]]:
    """Prepare all batch items from multiple files."""
    all_batch_items = []
    
    for qa_file in qa_files:
        # Extract table_id from filename
        table_id = qa_file.stem.replace("_qa", "")
        table_file = tables_dir / f"{table_id}.json"
        
        if not table_file.exists():
            cprint(f"Warning: Table file not found: {table_file}", "yellow")
            continue
        
        # Load table data
        table_data = load_table_data(table_file)
        
        # Load QA pairs and match with model responses
        qa_items = load_qa_pairs_with_responses(qa_file, model_responses)
        
        # Prepare batch items
        for qa_item in qa_items:
            if qa_item['question_id'] in processed_questions:
                continue
            
            all_batch_items.append({
                "question_id": qa_item["question_id"],
                "question": qa_item["question"],
                "question_type": qa_item["question_type"],
                "golden_answer": qa_item["answer"],
                "model_responses": qa_item["model_responses"],
                "table_data": table_data,
                "source_file": qa_file.name
            })
    
    return all_batch_items


def process_verifications(
    judge: VLLMGoldenAnswerJudge,
    qa_dir: Path,
    tables_dir: Path,
    model_response_files: List[str],
    output_file: str,
    batch_size: int = 32,
    parallel_files: int = 6
):
    """Process all QA pairs for golden answer verification with parallel file processing."""
    
    # Load model responses
    cprint("\nLoading model responses...", "blue")
    model_responses = load_model_responses(model_response_files)
    cprint(f"Loaded responses for {len(model_responses)} questions", "green")
    
    # Load already processed questions
    processed_questions = load_processed_questions(output_file)
    if processed_questions:
        cprint(f"Found {len(processed_questions)} already processed questions, resuming...", "green")
    
    qa_files = sorted(qa_dir.glob("*_qa.json"))
    cprint(f"\nFound {len(qa_files)} QA files to process", "white")
    cprint(f"Processing {parallel_files} files simultaneously in batches of {batch_size}", "cyan")
    
    # Prepare all batch items from multiple files at once
    cprint("\nPreparing batches from multiple files...", "blue")
    all_batch_items = prepare_file_batches(
        qa_files=qa_files,
        tables_dir=tables_dir,
        model_responses=model_responses,
        processed_questions=processed_questions,
        batch_size=batch_size
    )
    
    if not all_batch_items:
        cprint("All questions already processed!", "green")
        return
    
    cprint(f"Prepared {len(all_batch_items)} questions to verify", "white")
    
    total_processed = len(processed_questions)
    stats = {
        "correct": 0,
        "incorrect": 0,
        "needs_revision": 0,
        "errors": 0
    }
    
    # Process all items in batches
    total_batches = (len(all_batch_items) + batch_size - 1) // batch_size
    cprint(f"\nProcessing {total_batches} batches...", "blue")
    
    for i in range(0, len(all_batch_items), batch_size):
        batch = all_batch_items[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        # Show which files are in this batch
        source_files = set(item['source_file'] for item in batch)
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"Batch {batch_num}/{total_batches} ({len(batch)} items)", "cyan")
        cprint(f"Files in batch: {', '.join(sorted(source_files))}", "white")
        cprint(f"{'='*60}", "cyan")
        
        results = judge.generate_verification_batch(batch)
        
        # Save each result and update stats
        for result in results:
            judge.save_result_to_jsonl(result, output_file)
            total_processed += 1
            processed_questions.add(result["question_id"])
            
            # Update stats
            verdict = result["is_golden_correct"]
            if verdict == "Correct":
                stats["correct"] += 1
            elif verdict == "Incorrect":
                stats["incorrect"] += 1
            elif verdict == "Needs_Revision":
                stats["needs_revision"] += 1
            else:
                stats["errors"] += 1
        
        cprint(f"\n✓ Saved batch to {output_file} (Total: {total_processed})", "green")
        cprint(f"  Current Stats - Correct: {stats['correct']}, Incorrect: {stats['incorrect']}, Needs Revision: {stats['needs_revision']}", "white")
    
    # Print final summary
    cprint(f"\n{'='*60}", "blue")
    cprint(f"VERIFICATION COMPLETE", "blue")
    cprint(f"{'='*60}", "blue")
    cprint(f"\nTotal Questions Verified: {total_processed}", "white")
    cprint(f"  ✓ Correct Golden Answers: {stats['correct']} ({stats['correct']/total_processed*100:.1f}%)", "green")
    cprint(f"  ✗ Incorrect Golden Answers: {stats['incorrect']} ({stats['incorrect']/total_processed*100:.1f}%)", "red")
    cprint(f"  ⚠ Needs Revision: {stats['needs_revision']} ({stats['needs_revision']/total_processed*100:.1f}%)", "yellow")
    cprint(f"  ! Errors: {stats['errors']}", "red")
    
    accuracy_rate = (stats['correct'] / total_processed * 100) if total_processed > 0 else 0
    issue_rate = ((stats['incorrect'] + stats['needs_revision']) / total_processed * 100) if total_processed > 0 else 0
    
    cprint(f"\nGolden Answer Accuracy Rate: {accuracy_rate:.1f}%", "green" if accuracy_rate > 90 else "yellow")
    cprint(f"Issues Requiring Attention: {issue_rate:.1f}%", "red" if issue_rate > 10 else "yellow")


def main():
    """Main execution function."""
    
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Thinking"
    TENSOR_PARALLEL_SIZE = 4
    GPU_MEMORY_UTILIZATION = 0.8
    MAX_MODEL_LEN = 32768
    ENABLE_THINKING = True
    USE_GUIDED_DECODING = True
    BATCH_SIZE = 16 # Number of questions per batch
    PARALLEL_FILES = 6  # Number of files to process simultaneously
    
    # Set up paths
    base_dir = Path("/home/anshulsc/links/scratch/projects/MMTQA/data/processed/")
    qa_dir = base_dir / "qa_pairs"
    tables_dir = base_dir / "tables"
    
    # Model response files (3 JSONL files from different models)
    model_response_files = [
        base_dir / "evaluation_results" / "OpenGVLab_InternVL3-38B-Instruct_multitableqa_clean_en_20251020_144340.jsonl",
        base_dir / "evaluation_results" / "Qwen_Qwen2.5-VL-72B-Instruct_multitableqa_clean_en_20251020_131858.jsonl"
    ]
    
    output_file = base_dir / "golden_answer_verification_results_v5CU.jsonl"
    
    # Verify directories exist
    if not qa_dir.exists():
        raise FileNotFoundError(f"QA directory not found: {qa_dir}")
    if not tables_dir.exists():
        raise FileNotFoundError(f"Tables directory not found: {tables_dir}")
    
    cprint("="*60, "blue")
    cprint("GOLDEN ANSWER VERIFICATION SYSTEM (PARALLEL MODE)", "blue")
    cprint("="*60, "blue")
    cprint(f"\nQA directory: {qa_dir}", "white")
    cprint(f"Tables directory: {tables_dir}", "white")
    cprint(f"Model response files: {len(model_response_files)}", "white")
    cprint(f"Batch size: {BATCH_SIZE}", "white")
    cprint(f"Parallel files: {PARALLEL_FILES}", "white")
    cprint(f"Output file: {output_file}\n", "white")
    
    # Initialize judge
    judge = VLLMGoldenAnswerJudge(
        model_name=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        enable_thinking=ENABLE_THINKING,
        use_guided_decoding=USE_GUIDED_DECODING
    )
    
    # Process all verifications
    process_verifications(
        judge=judge,
        qa_dir=qa_dir,
        tables_dir=tables_dir,
        model_response_files=model_response_files,
        output_file=output_file,
        batch_size=BATCH_SIZE,
        parallel_files=PARALLEL_FILES
    )


if __name__ == "__main__":
    main()