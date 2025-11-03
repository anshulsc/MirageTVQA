import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


INPUT_QA_DIR = PROCESSED_DATA_DIR / "qa_pairs"
TABLES_DIR = PROCESSED_DATA_DIR / "tables" 

OUTPUT_DIR = PROCESSED_DATA_DIR / "qa_pairs_translated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUESTS_PER_MINUTE = 10 


LANGUAGES = {
    # Afro-Asiatic
    "ar": "Arabic (MSA)",
    "he": "Hebrew",
    "am": "Amharic",
    
    # Austronesian
    "id_casual": "Indonesian (Casual)",
    "id_formal": "Indonesian (Formal)",
    "jv_krama": "Javanese (Krama - Polite)",
    "jv_ngoko": "Javanese (Ngoko - Casual)",
    "su_loma": "Sundanese",
    "tl": "Tagalog",
    "ms": "Malay",
    "fil": "Filipino",
    
    # Indo-European
    "bn": "Bengali",
    "cs": "Czech",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "mr": "Marathi",
    "ru_formal": "Russian (Formal)",
    "sc": "Sardinian",
    "si_formal_spoken": "Sinhala",
    "fa": "Persian",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "pl": "Polish",
    "no": "Norwegian",
    "sv": "Swedish",
    "da": "Danish",
    "el": "Greek",
    "ur": "Urdu",
    "pb": "Punjabi",
    "np": "Nepali",
    "pt": "Portuguese",
    
    # Japonic
    "ja_formal": "Japanese (Formal)",
    
    # Koreanic
    "ko_formal": "Korean (Formal)",
    
    # Kra-Dai
    "th": "Thai",
    
    # Sino-Tibetan
    "nan": "Hokkien (Written)",
    "zh_cn": "Chinese (Mandarin)",
    "my": "Burmese",
    
    # Turkic
    "az": "Azerbaijani",
    "tr": "Turkish",
    
    # Austroasiatic
    "vi": "Vietnamese",
    
    # Dravidian
    "ta": "Tamil",
    "te": "Telugu",
}


GEMINI_API_KEYS = [
"API"
]
GEMINI_MODEL_NAME = "gemini-2.5-flash"