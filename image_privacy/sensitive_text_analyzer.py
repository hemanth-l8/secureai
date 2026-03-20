import re
import logging
from .utils import apply_pixelation

# Import NER from security middleware
try:
    from security.middleware.ner import NERMiddleware
except ImportError:
    try:
        from middleware.ner import NERMiddleware
    except ImportError:
        NERMiddleware = None

logger = logging.getLogger(__name__)

class SensitiveTextAnalyzer:
    """
    Advanced text analysis for privacy. Uses Hybrid NER (ML + Regex).
    """
    def __init__(self):
        self.ner_layer = NERMiddleware() if NERMiddleware else None
        
        # Fallback Keywords
        self.sensitive_keywords = ["NAME", "ID", "TEL", "DOB", "EXP", "ADDRESS", "GENDER", "ISSUE", "BLOOD", "PRINCIPAL", "B.TECH", "ROLL", "INSTITUTE"]
        
        # Regex Patterns (Fallback)
        self.patterns = {
            "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "id_num": r'\b(?:ID|ROLL|REG)\s*[:.]?\s*[A-Z0-9-]{5,}\b',
            "alphanumeric": r'\b[A-Z0-9]{8,12}\b'
        }

    def analyze_and_mask(self, image, ocr_results):
        """
        Scans OCR results for sensitive patterns using Hybrid NER.
        """
        detected_sensitive_text = []
        if not ocr_results:
            return image, []
            
        full_text = " ".join([item['text'] for item in ocr_results])
        
        # 1. Use NER scanner if available
        detected_entities = []
        if self.ner_layer:
            ner_results = self.ner_layer.scan(full_text)
            detected_entities = ner_results.get('entities', [])
            
        sensitive_spans = [(e['text'].lower(), e['label']) for e in detected_entities]
        
        for item in ocr_results:
            text = item['text']
            text_lower = text.lower()
            text_upper = text.upper()
            box = item['box']
            should_mask = False
            found_types = []
            
            # Policy 1: Hybrid NER Match
            for s_text, s_label in sensitive_spans:
                if s_text in text_lower or text_lower in s_text:
                    should_mask = True
                    found_types.append(s_label)
                    break
            
            # Policy 2: Keywords & Regex Fallback
            if not should_mask:
                if any(kw in text_upper for kw in self.sensitive_keywords):
                    should_mask = True
                    found_types.append("SENSITIVE_KEYWORD")
                else:
                    for label, pattern in self.patterns.items():
                        if re.search(pattern, text):
                            should_mask = True
                            found_types.append(label.upper())
                            break
            
            if should_mask:
                image = apply_pixelation(image, box)
                detected_sensitive_text.append({
                    "text_snippet": text,
                    "types": found_types,
                    "box": box
                })
                
        return image, detected_sensitive_text
