To switch methods, edit text_extraction_config.py:

pythonOCR_METHOD = 'deepseek'  # or 'docling' or 'tesseract'

Then run:

python -m text_extraction/run_extraction.py

That's it! The system automatically uses the configured method.