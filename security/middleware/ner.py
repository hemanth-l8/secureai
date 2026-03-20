
import re
import logging
from utils.patterns import PATTERNS, CONTEXTUAL_PATTERNS

try:
    import spacy
    NLP_MODEL = None
except ImportError:
    spacy = None
    NLP_MODEL = None

class NERMiddleware:
    """
    Hybrid Named Entity Recognition Middleware.
    Combines Regex (Structured Data) + ML/SpaCy (Contextual Entities).
    """

    def __init__(self, spacy_model_name="en_core_web_sm"):
        self.patterns = PATTERNS
        self.contextual_patterns = CONTEXTUAL_PATTERNS
        self.nlp = None
        
        # Initialize SpaCy if available
        if spacy:
            try:
                logging.info(f"NER: Loading SpaCy model '{spacy_model_name}'...")
                self.nlp = spacy.load(spacy_model_name)
                logging.info("NER: SpaCy model loaded successfully.")
            except OSError:
                logging.warning(f"NER: SpaCy model '{spacy_model_name}' not found. Please run: python -m spacy download {spacy_model_name}")
                self.nlp = None
        else:
            logging.warning("NER: SpaCy library not installed. Falling back to Regex-only detection.")

    def detect_regex_entities(self, text):
        """Scans for structured data (Aadhaar, PAN, Emails, etc). Returns list of (start, end, label, text)."""
        detected = []
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                detected.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": label,
                    "text": match.group()
                })
        return detected, len(detected)

    def detect_contextual_entities(self, text):
        """Scans for sensitive data that requires surrounding context. Returns list of (start, end, label, text)."""
        detected = []
        blacklist = ["here", "ready", "fine", "good", "busy", "there", "back"]
        
        for label, pattern in self.contextual_patterns.items():
            for match in re.finditer(pattern, text):
                # For patterns with capture groups, we only mask the group content
                if match.groups():
                    val = match.group(1)
                    start, end = match.start(1), match.end(1)
                else:
                    val = match.group()
                    start, end = match.start(), match.end()
                
                if label == "NAME" and val.lower() in blacklist:
                    continue
                    
                detected.append({
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": val
                })
                
        return detected, len(detected)

    def detect_ml_entities(self, text):
        """Scans for contextual entities (PERSON, ORG, GPE) using SpaCy with risk filtering."""
        detected = []
        high_risk_keywords = ["government", "police", "hospital", "bank", "court", "ministry", "agency"]
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                should_mask = False
                if ent.label_ == "PERSON":
                    # Basic validation: Usually names are > 2 chars
                    if len(ent.text) > 2:
                        should_mask = True
                elif ent.label_ in ["ORG", "GPE"]:
                    # Refinement: Only mask ORG/GPE if they contain high-risk keywords
                    if any(kw in ent.text.lower() for kw in high_risk_keywords):
                        should_mask = True
                
                if should_mask:
                    detected.append({
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": ent.label_,
                        "text": ent.text
                    })
                    
        return detected, len(detected)

    def scan(self, text):
        """
        Main entry point for scanning text.
        Returns a consolidated list of all entities with spans.
        """
        regex_entities, r_count = self.detect_regex_entities(text)
        context_entities, c_count = self.detect_contextual_entities(text)
        ml_entities, m_count = self.detect_ml_entities(text)
        
        # Merge all findings into a single list
        all_entities = regex_entities + context_entities + ml_entities
        
        # Sort by start position to facilitate span-based replacement
        all_entities.sort(key=lambda x: x['start'])
        
        return {
            "entities": all_entities,
            "total_sensitive_items": len(all_entities)
        }

def get_ner_module():
    return NERMiddleware()
