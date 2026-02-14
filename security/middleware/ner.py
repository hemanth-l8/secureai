
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
        """Scans for structured data (Aadhaar, PAN, Emails, etc)."""
        detected = {}
        total_count = 0
        
        for label, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                 # Flatten if necessary, but keep list of occurrences
                detected[label] = matches
                total_count += len(matches)
                
        return detected, total_count

    def detect_contextual_entities(self, text):
        """Scans for sensitive data that requires surrounding context (e.g. Usernames)."""
        detected = {}
        total_count = 0
        
        for label, pattern in self.contextual_patterns.items():
            # Use finditer to get capture groups
            matches = re.finditer(pattern, text)
            for match in matches:
                # We want the username, which is the first capture group
                username = match.group(1)
                
                # Blacklist for names to avoid over-detection (e.g. "i'm here")
                blacklist = ["here", "ready", "fine", "good", "busy", "there", "back"]
                if label == "NAME" and username.lower() in blacklist:
                    continue
                    
                if label not in detected:
                    detected[label] = []
                detected[label].append(username)
                total_count += 1
                
        return detected, total_count

    def detect_ml_entities(self, text):
        """Scans for contextual entities (PERSON, ORG, GPE) using SpaCy."""
        ml_entities = {}
        ml_count = 0
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                    if ent.label_ not in ml_entities:
                        ml_entities[ent.label_] = []
                    ml_entities[ent.label_].append(ent.text)
                    ml_count += 1
                    
        return ml_entities, ml_count

    def scan(self, text):
        """
        Main entry point for scanning text.
        Returns a consolidated report of all findings.
        """
        # 1. Regex Scan (Structured)
        regex_data, regex_count = self.detect_regex_entities(text)
        
        # 2. Contextual Regex Scan (Usernames, etc)
        contextual_regex_data, contextual_regex_count = self.detect_contextual_entities(text)
        
        # 3. ML Scan
        ml_data, ml_count = self.detect_ml_entities(text)
        
        # 4. Merge Results
        full_report = {
            "structured_data": regex_data,     
            "contextual_regex_data": contextual_regex_data, 
            "contextual_data": ml_data,        
            "total_sensitive_items": regex_count + contextual_regex_count + ml_count
        }
        
        return full_report

def get_ner_module():
    return NERMiddleware()
