import torch
import json
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import math

class AttentionOutputExtractor:
    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = config.DEVICE
        
        logging.basicConfig(
            level=logging.INFO if config.VERBOSE else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.model.eval()
        self.model.set_attn_implementation("eager") 
        
    def _create_visual_only_mask(self, batch_size, seq_length, num_image_tokens):
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=self.device), 
            diagonal=1
        )
        last_token_idx = seq_length - 1
        text_token_start_idx = num_image_tokens + 1
        causal_mask[last_token_idx, text_token_start_idx:] = True
        attention_mask = torch.zeros((batch_size, 1, seq_length, seq_length), device=self.device)
        attention_mask.masked_fill_(causal_mask.bool(), float('-inf'))
        return attention_mask

    def extract_attention_outputs(self, inputs, use_masked_attention):
        num_image_tokens = 576
        batch_size, seq_length = inputs["input_ids"].shape
        
        if use_masked_attention:
            custom_mask = self._create_visual_only_mask(batch_size, seq_length, num_image_tokens)
            inputs["attention_mask"] = custom_mask
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )

        attention_outputs_per_layer = []
        last_token_idx = inputs["input_ids"].shape[1] - 1

        for layer_idx in range(self.config.NUM_LAYERS):
            attn_weights = outputs.attentions[layer_idx]
            hidden_states = outputs.hidden_states[layer_idx]
            
            v_proj = self.model.model.language_model.layers[layer_idx].self_attn.v_proj
            value_states = v_proj(hidden_states)
            value_states = value_states.view(
                batch_size, seq_length, self.config.NUM_HEADS, self.config.HEAD_DIM
            ).transpose(1, 2)
            attention_outputs = torch.matmul(attn_weights, value_states)
            last_token_head_outputs = attention_outputs[0, :, last_token_idx, :]
            attention_outputs_per_layer.append(last_token_head_outputs)
            
        return attention_outputs_per_layer

    def process_dataset(self, dataset, images_base_path):
        unique_labels = list(set(record["probe_label"] for record in dataset))
        
        if len(unique_labels) != 2:
            raise ValueError(f"Expected exactly 2 unique probe_labels, but found {len(unique_labels)}: {unique_labels}")
        
        label_a, label_b = sorted(unique_labels)  # Sort for consistency
        
        self.logger.info(f"Starting feature extraction for {label_a} vs {label_b}")
        self.logger.info(f"Images base path: {images_base_path}")

        file_handles = {}
        for l in range(self.config.NUM_LAYERS):
            for h in range(self.config.NUM_HEADS):
                filepath = self.config.FEATURES_DIR / f"features_layer_{l}_head_{h}.jsonl"
                if filepath.exists(): 
                    filepath.unlink()
                file_handles[(l, h)] = open(filepath, "a")

        images_base = Path(images_base_path)
        
        for i in tqdm(range(len(dataset)), desc="Extracting features"):
            try:
                record = dataset[i]
                table_id = record["table_id"]
                question = record["question"]
                probe_label = record["probe_label"]
                
                if probe_label == label_a:
                    label = 1
                else:  # probe_label == label_b
                    label = -1
                
                prompt = self.config.PROMPT_TEMPLATE.format(question=question)
                
                image_path = images_base / table_id / "clean" / "en_clean.jpg"
                
                if not image_path.exists():
                    self.logger.warning(f"Image not found for sample {i}: {image_path}")
                    continue
                
                image = Image.open(image_path)
                
                if image.mode != "RGB": 
                    image = image.convert("RGB")

                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                
                outputs_per_layer = self.extract_attention_outputs(
                    inputs, 
                    use_masked_attention=self.config.USE_MASKED_ATTENTION_FOR_PROBES
                )
                
                for layer_idx in range(self.config.NUM_LAYERS):
                    head_outputs = outputs_per_layer[layer_idx]
                    for head_idx in range(self.config.NUM_HEADS):
                        feature_vector = head_outputs[head_idx].cpu().detach().numpy().tolist()
                        record_data = {
                            "image_idx": i,
                            "table_id": table_id,
                            "question": question,
                            "feature": feature_vector,
                            "label": label,
                            "probe_label": probe_label
                        }
                        fh = file_handles[(layer_idx, head_idx)]
                        fh.write(json.dumps(record_data) + "\n")
            
            except FileNotFoundError as e:
                self.logger.warning(f"Image file not found for sample {i}: {e}")
            except Exception as e:
                self.logger.error(f"Error processing sample {i}: {e}", exc_info=True)

        for fh in file_handles.values(): 
            fh.close()
        self.logger.info(f"Feature extraction complete. Data saved to {self.config.FEATURES_DIR}")


class Qwen25VLAttentionOutputExtractor:
    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = config.DEVICE
        
        logging.basicConfig(
            level=logging.INFO if config.VERBOSE else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Model structure: {type(self.model)}")
        self.model.eval()
        
        self.num_key_value_heads = self.model.model.language_model.layers[0].self_attn.num_key_value_heads
        self.num_query_heads = self.config.NUM_HEADS
        self.logger.info(f"Model uses GQA: {self.num_query_heads} query heads, {self.num_key_value_heads} KV heads")

    def _get_vision_token_indices(self, input_ids):
        special_tokens_map = {
            "<|vision_start|>": None,
            "<|image_pad|>": None,
            "<|vision_end|>": None,
        }
        
        for token_name in special_tokens_map.keys():
            try:
                token_id = self.processor.tokenizer.convert_tokens_to_ids(token_name)
                if token_id != self.processor.tokenizer.unk_token_id:
                    special_tokens_map[token_name] = token_id
            except:
                pass
        
        vision_indices = []
        in_vision = False
        
        for idx, token_id in enumerate(input_ids[0].tolist()):
            if special_tokens_map.get("<|vision_start|>") == token_id:
                in_vision = True
                vision_indices.append(idx)
            elif special_tokens_map.get("<|vision_end|>") == token_id:
                vision_indices.append(idx)
                in_vision = False
            elif in_vision or token_id == special_tokens_map.get("<|image_pad|>"):
                vision_indices.append(idx)
        
        # Fallback heuristic
        if not vision_indices:
            self.logger.warning("Using heuristic for vision token detection")
            for idx, token_id in enumerate(input_ids[0].tolist()):
                if token_id >= 151643:
                    vision_indices.append(idx)
        
        return vision_indices

    def _compute_attention_manually(self, layer, hidden_states, vision_token_indices, use_masked_attention):
        batch_size, seq_length, hidden_size = hidden_states.shape
        last_token_idx = seq_length - 1
        
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        
        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_length, self.num_query_heads, self.config.HEAD_DIM).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.config.HEAD_DIM).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.config.HEAD_DIM).transpose(1, 2)
        
        key_states = key_states.repeat_interleave(self.num_query_heads // self.num_key_value_heads, dim=1)
        value_states = value_states.repeat_interleave(self.num_query_heads // self.num_key_value_heads, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.config.HEAD_DIM)
        
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), device=hidden_states.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if use_masked_attention and len(vision_token_indices) > 0:
            mask = torch.ones(seq_length, dtype=torch.bool, device=hidden_states.device)
            mask[vision_token_indices] = False
            attn_weights[:, :, last_token_idx, mask] = float('-inf')
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        last_token_head_outputs = attn_output[0, :, last_token_idx, :] 
        
        return last_token_head_outputs

    def extract_attention_outputs(self, inputs, use_masked_attention):
        batch_size, seq_length = inputs["input_ids"].shape
        vision_token_indices = self._get_vision_token_indices(inputs["input_ids"])
        
        with torch.no_grad():
            outputs = self.model.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        attention_outputs_per_layer = []
        
        for layer_idx in range(self.config.NUM_LAYERS):
            layer = self.model.model.language_model.layers[layer_idx]
            hidden_states = outputs.hidden_states[layer_idx]
            
            last_token_outputs = self._compute_attention_manually(
                layer, 
                hidden_states, 
                vision_token_indices, 
                use_masked_attention
            )
            
            attention_outputs_per_layer.append(last_token_outputs)
            
        return attention_outputs_per_layer

    def process_dataset(self, dataset, images_base_path):
        """Process custom JSON dataset with table_id, question, and probe_label"""
        unique_labels = list(set(record["probe_label"] for record in dataset))
        
        if len(unique_labels) != 2:
            raise ValueError(f"Expected exactly 2 unique probe_labels, but found {len(unique_labels)}: {unique_labels}")
        
        label_a, label_b = sorted(unique_labels)  # Sort for consistency
        
        self.logger.info(f"Starting feature extraction for {label_a} vs {label_b}")
        self.logger.info(f"Images base path: {images_base_path}")

        file_handles = {}
        for l in range(self.config.NUM_LAYERS):
            for h in range(self.config.NUM_HEADS):
                filepath = self.config.FEATURES_DIR / f"features_layer_{l}_head_{h}.jsonl"
                if filepath.exists(): 
                    filepath.unlink()
                file_handles[(l, h)] = open(filepath, "a")

        images_base = Path(images_base_path)
        
        for i in tqdm(range(len(dataset)), desc="Extracting features"):
            try:
                record = dataset[i]
                table_id = record["table_id"]
                question = record["question"]
                probe_label = record["probe_label"]
                
                if probe_label == label_a:
                    label = 1
                else: 
                    label = -1
                
                # image_path = images_base / table_id / "clean" / "en_clean.jpg"
                image_path = Path(record["image_path"])
                
                if not image_path.exists():
                    self.logger.warning(f"Image not found for sample {i}: {image_path}")
                    continue
                
                image = Image.open(image_path)
                
                if image.mode != "RGB": 
                    image = image.convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": " "}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs_per_layer = self.extract_attention_outputs(
                    inputs, 
                    use_masked_attention=self.config.USE_MASKED_ATTENTION_FOR_PROBES
                )
                
                for layer_idx in range(self.config.NUM_LAYERS):
                    head_outputs = outputs_per_layer[layer_idx]
                    for head_idx in range(self.config.NUM_HEADS):
                        feature_vector = head_outputs[head_idx].cpu().detach().numpy().tolist()
                        record_data = {
                            "image_idx": i,
                            "table_id": table_id,
                            "question": question,
                            "feature": feature_vector,
                            "label": label,
                            "probe_label": probe_label
                        }
                        fh = file_handles[(layer_idx, head_idx)]
                        fh.write(json.dumps(record_data) + "\n")
            
            except FileNotFoundError as e:
                self.logger.warning(f"Image file not found for sample {i}: {e}")
            except Exception as e:
                self.logger.error(f"Error processing sample {i}: {e}", exc_info=True)

        for fh in file_handles.values(): 
            fh.close()
            
        self.logger.info(f"Feature extraction complete. Data saved to {self.config.FEATURES_DIR}")