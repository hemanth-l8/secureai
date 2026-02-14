import re
from .utils import apply_pixelation

class SensitiveTextAnalyzer:
    """
    Analyzes extracted text for sensitive patterns and masks them using pixelation.
    """
    def __init__(self):
        # Define regex patterns for sensitive data
        self.patterns = {
            "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "aadhaar": r'\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b',
            "pan": r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            "credit_card": r'\b(?:\d[ -]*?){13,16}\b'
        }

    def analyze_and_mask(self, image, ocr_results):
        """
        Scans OCR results for sensitive patterns.
        Masks the corresponding image regions and returns detected categories.
        """
        detected_sensitive_text = []
        
        for item in ocr_results:
            text = item['text']
            box = item['box']
            is_sensitive = False
            found_types = []
            
            for label, pattern in self.patterns.items():
                if re.search(pattern, text):
                    is_sensitive = True
                    found_types.append(label)
            
            if is_sensitive:
                image = apply_pixelation(image, box)
                detected_sensitive_text.append({
                    "text_snippet": text,
                    "types": found_types,
                    "box": box
                })
                
        return image, detected_sensitive_text
