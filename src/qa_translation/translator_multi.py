import json
import time
import os
from typing import Dict, Any, Optional, Set
from collections import deque
from datetime import datetime, timedelta
from termcolor import cprint
import google.generativeai as genai
from pathlib import Path

from src.configs import qa_translation_config as cfg
from .prompts import QA_TRANSLATION_PROMPT, TranslatedQA


class APIKeyRotator:
    """Process-safe API key manager - each worker gets assigned a specific starting key"""
    
    def __init__(self, api_keys: list, requests_per_minute: int = 12, worker_id: int = None):
        self.api_keys = api_keys
        self.requests_per_minute = requests_per_minute
        
        # Assign key based on worker ID (preferred) or process ID as fallback
        if worker_id is not None:
            self.worker_id = worker_id
        else:
            # Fallback to process ID if worker_id not provided
            self.worker_id = os.getpid() % 1000  # Use last 3 digits of PID
        
        self.current_key_index = self._get_initial_key_index()
        
        # Track request count for current key
        self.current_key_request_count = 0
        self.last_request_time = time.time()
        
        # Track keys that have exceeded quota
        self.quota_exceeded_keys: Set[int] = set()
        
        cprint(f"    [Worker #{self.worker_id}] Initialized with API Key #{self.current_key_index + 1}", "cyan")
        
    def _get_initial_key_index(self) -> int:
        """Get initial key index based on worker ID"""
        # Use modulo to distribute workers across available keys
        # Worker 1 -> Key 0, Worker 2 -> Key 1, etc.
        return (self.worker_id - 1) % len(self.api_keys)
    
    def get_current_key(self) -> str:
        """Get the current API key"""
        return self.api_keys[self.current_key_index]
    
    def get_current_key_index(self) -> int:
        """Get the current key index for logging"""
        return self.current_key_index
    
    def mark_quota_exceeded(self, key_index: int):
        """Mark a key as having exceeded quota and rotate to next available key"""
        self.quota_exceeded_keys.add(key_index)
        cprint(f"    [Worker #{self.worker_id}] Key #{key_index + 1} quota exceeded. "
               f"{len(self.quota_exceeded_keys)}/{len(self.api_keys)} keys exhausted.", "red")
        
        if self.has_available_keys():
            self._rotate_to_next_available_key()
    
    def has_available_keys(self) -> bool:
        """Check if there are any keys left that haven't exceeded quota"""
        return len(self.quota_exceeded_keys) < len(self.api_keys)
    
    def _rotate_to_next_available_key(self):
        """Rotate to the next key that hasn't exceeded quota"""
        attempts = 0
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
            
            if self.current_key_index not in self.quota_exceeded_keys:
                self.current_key_request_count = 0
                cprint(f"    [Worker #{self.worker_id}] Switched to API key #{self.current_key_index + 1}", "cyan")
                return
        
        raise Exception(f"Worker #{self.worker_id}: All API keys have exceeded their quota.")
    
    def check_and_wait_if_needed(self):
        """Check if we need to wait based on rate limiting"""
        current_time = time.time()
        time_since_last_batch = current_time - self.last_request_time
        
        if self.current_key_request_count >= self.requests_per_minute:
            # If we've made requests_per_minute requests and less than 60s has passed
            wait_time = 30  # Always wait 30 seconds after batch
            cprint(f"    [Worker #{self.worker_id}] Rate limit reached on key #{self.current_key_index + 1}. "
                   f"Waiting {wait_time}s...", "yellow")
            time.sleep(wait_time)
            
            # Reset counter
            self.current_key_request_count = 0
            self.last_request_time = time.time()
            cprint(f"    [Worker #{self.worker_id}] Resumed with key #{self.current_key_index + 1}", "green")
    
    def record_request(self):
        """Record that a request was made with the current key"""
        self.current_key_request_count += 1


class TranslationTracker:
    """Tracks completed translations - no longer needed with multiprocessing Manager"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.completed_cache: Dict[str, Set[str]] = {}
    
    def _get_cache_key(self, lang_code: str) -> str:
        return lang_code
    
    def is_completed(self, lang_code: str, table_id: str) -> bool:
        cache_key = self._get_cache_key(lang_code)
        
        if cache_key not in self.completed_cache:
            self.completed_cache[cache_key] = self._load_completed_files(lang_code)
        
        return table_id in self.completed_cache[cache_key]
    
    def _load_completed_files(self, lang_code: str) -> Set[str]:
        lang_dir = self.output_dir / lang_code
        if not lang_dir.exists():
            return set()
        
        completed = set()
        for json_file in lang_dir.glob("*.json"):
            table_id = json_file.stem.replace("_qa", "")
            completed.add(table_id)
        
        return completed
    
    def mark_completed(self, lang_code: str, table_id: str):
        cache_key = self._get_cache_key(lang_code)
        if cache_key not in self.completed_cache:
            self.completed_cache[cache_key] = set()
        self.completed_cache[cache_key].add(table_id)


class QATranslator:
    """Translates QA pairs - each instance gets its own API key rotator"""
    
    def __init__(self, english_qa_pair: Dict[str, Any], context_table: Dict[str, Any], 
                 table_id: str = "unknown", worker_id: int = None):
        self.english_qa_pair = english_qa_pair
        self.context_table = context_table
        self.table_id = table_id
        
        # Each worker gets its own key rotator with assigned starting key
        self.key_rotator = APIKeyRotator(
            api_keys=cfg.GEMINI_API_KEYS,
            requests_per_minute=getattr(cfg, 'REQUESTS_PER_MINUTE', 10),
            worker_id=worker_id
        )
        
        self.model = None
        self._configure_model()
    
    def _configure_model(self):
        """Configure Gemini model with current API key"""
        current_key = self.key_rotator.get_current_key()
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel(
            cfg.GEMINI_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}
        )

    def translate(self, target_language: str, max_retries: int = 3) -> Optional[TranslatedQA]:
        """
        Translate the QA pair to the target language with automatic key rotation
        
        Args:
            target_language: Name of target language (e.g., "Spanish", "French")
            max_retries: Maximum number of retry attempts per error
            
        Returns:
            TranslatedQA object or None if translation fails
        """
        # Prepare the context table and QA pair as JSON strings
        context_table_json = json.dumps(self.context_table, indent=2, ensure_ascii=False)
        english_qa_json = json.dumps({
            "question": self.english_qa_pair.get("question", ""),
            "answer": self.english_qa_pair.get("answer", []),
            "question_type": self.english_qa_pair.get("question_type", "value")
        }, indent=2, ensure_ascii=False)
        
        # Format the prompt
        prompt = QA_TRANSLATION_PROMPT.format(
            target_language=target_language,
            context_table_json=context_table_json,
            english_qa_json=english_qa_json
        )
        
        total_attempts = 0
        max_total_attempts = max_retries * len(cfg.GEMINI_API_KEYS)
        
        while total_attempts < max_total_attempts:
            try:
                # Check if all keys are exhausted
                if not self.key_rotator.has_available_keys():
                    cprint(f"    [Worker #{self.key_rotator.worker_id}] All API keys exhausted!", "red")
                    return None
                
                # Check and wait if needed
                self.key_rotator.check_and_wait_if_needed()
                
                key_index = self.key_rotator.get_current_key_index()
                
                # Generate translation
                response = self.model.generate_content(prompt)
                
                # Record successful request
                self.key_rotator.record_request()
                
                # Validate the response
                validated_translation = TranslatedQA.model_validate_json(response.text)
                
                cprint(f"      [{self.table_id}] Translation to {target_language} completed "
                       f"(Key #{key_index + 1}, Request #{self.key_rotator.current_key_request_count})", "green")
                
                return validated_translation

            except Exception as e:
                error_msg = str(e).lower()
                total_attempts += 1
                key_index = self.key_rotator.get_current_key_index()
                
                # Handle quota exceeded
                if any(keyword in error_msg for keyword in ["quota exceeded", "resource exhausted", "429"]):
                    self.key_rotator.mark_quota_exceeded(key_index)
                    
                    if not self.key_rotator.has_available_keys():
                        return None
                    
                    # Reconfigure model with new key
                    self._configure_model()
                    time.sleep(getattr(cfg, 'QUOTA_ERROR_DELAY', 2))
                
                # Handle rate limit errors
                elif "rate limit" in error_msg:
                    time.sleep(60)
                    self.key_rotator.current_key_request_count = 0
                
                # Handle other errors with backoff
                elif total_attempts < max_total_attempts:
                    wait_time = min(2 ** (total_attempts % 5), 16)
                    time.sleep(wait_time)
        
        return None
    
    @classmethod
    def should_skip_translation(cls, lang_code: str, table_id: str) -> bool:
        """Check if translation should be skipped (already completed)"""
        output_dir = cfg.OUTPUT_DIR / lang_code
        output_file = output_dir / f"{table_id}_qa.json"
        return output_file.exists()
    
    @classmethod
    def mark_translation_complete(cls, lang_code: str, table_id: str):
        """Mark translation as complete - no-op in multiprocessing version"""
        pass