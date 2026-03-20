import re
import math
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for Classification
SENSITIVITY_THRESHOLD = 0.45  # Confidence score above which we flag as sensitive document

# Document Scoring Weights and Patterns
# These are CPU-friendly patterns and keyword weights for classification
DOCUMENT_PATTERNS = {
    "AADHAAR": {
        "keywords": ["AADHAAR", "UIDAI", "GOVERNMENT OF INDIA", "MALE", "FEMALE", "YEAR OF BIRTH", "DOB", "ENROLMENT NO", "VID"],
        "patterns": [
            r"\d{4}\s\d{4}\s\d{4}",  # Aadhaar Number format
            r"HELP@UIDAI.GOV.IN",
            r"WWW.UIDAI.GOV.IN"
        ],
        "weight": 1.2
    },
    "PAN": {
        "keywords": ["INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "PERMANENT ACCOUNT NUMBER", "CARD", "SIGNATURE"],
        "patterns": [
            r"[A-Z]{5}[0-9]{4}[A-Z]"  # PAN Number format
        ],
        "weight": 1.1
    },
    "PASSPORT": {
        "keywords": ["REPUBLIC OF INDIA", "PASSPORT", "P-INDIA", "NATIONALITY", "PLACE OF BIRTH", "DATE OF ISSUE", "DATE OF EXPIRY"],
        "patterns": [
            r"[A-Z][0-9]{7}"  # Passport Number format
        ],
        "weight": 1.3
    },
    "DRIVER_LICENSE": {
        "keywords": ["DRIVER", "LICENSE", "TRANSPORT DEPARTMENT", "AUTHORITY", "DL NO", "VALID UPTO"],
        "patterns": [
            r"[A-Z]{2}[0-9]{2}\s?[0-9]{11}"  # DL Number format
        ],
        "weight": 1.1
    },
    "BANK_CARD": {
        "keywords": ["DEBIT", "CREDIT", "VISA", "MASTERCARD", "RUPAY", "HDFC", "SBI", "ICICI", "AXIS"],
        "patterns": [
            r"([0-9]{4}\s?){4}"  # 16-digit card number
        ],
        "weight": 1.0
    },
    "SALARY_SLIP": {
        "keywords": ["SALARY", "SLIP", "EARNINGS", "DEDUCTIONS", "NET PAY", "PF", "BASIC", "HRA", "LTA"],
        "patterns": [
            r"SALARY\s?SLIP",
            r"MONTHLY\s?PAY"
        ],
        "weight": 0.9
    },
    "MEDICAL": {
        "keywords": ["PATIENT", "DIAGNOSIS", "HOSPITAL", "CLINIC", "PRESCRIPTION", "REPORT", "LAB", "BLOOD TEST"],
        "patterns": [
            r"PATIENT\s?NAME",
            r"RX\s?"
        ],
        "weight": 1.0
    },
    "CONFIDENTIAL": {
        "keywords": ["CONFIDENTIAL", "PRIVATE", "PROPRIETARY", "INTERNAL ONLY", "NON-DISCLOSURE", "AGREEMENT", "TRADE SECRET"],
        "patterns": [
            r"STRICTLY\s?CONFIDENTIAL"
        ],
        "weight": 1.4
    }
}

# PART 6: Document-Specific Masking Rules
DOCUMENT_MASKING_RULES = {
    "AADHAAR": {
        "mask_all_text": True,
        "mask_faces": True,
        "mask_qr": True,
        "risk_escalation": 0.5
    },
    "PAN": {
        "mask_all_text": True,
        "mask_faces": False,
        "mask_qr": False,
        "risk_escalation": 0.4
    },
    "PASSPORT": {
        "mask_all_text": True,
        "mask_faces": True,
        "mask_qr": False,
        "risk_escalation": 0.6
    },
    "BANK_CARD": {
        "mask_all_text": True,
        "mask_faces": False,
        "mask_qr": False,
        "risk_escalation": 0.7
    },
    "CONFIDENTIAL": {
        "mask_all_text": False,
        "mask_faces": False,
        "mask_qr": False,
        "risk_escalation": 0.4
    },
    "NONE": {
        "mask_all_text": False,
        "mask_faces": False,
        "mask_qr": False,
        "risk_escalation": 0.0
    }
}

class DocumentClassifier:
    """
    Lightweight, pattern-based classifier for sensitive documents.
    Optimized for CPU usage and fast inference.
    """
    def __init__(self):
        self.patterns = DOCUMENT_PATTERNS

    def classify(self, text: str) -> Dict:
        """Determines the document type and confidence score based on text content."""
        if not text:
            return {"document_type": "NONE", "confidence": 0.0, "is_sensitive_document": False}

        text_upper = text.upper()
        scores = {}

        for doc_type, config in self.patterns.items():
            keyword_match_count = sum(1 for kw in config["keywords"] if kw in text_upper)
            pattern_match_count = sum(1 for pat in config["patterns"] if re.search(pat, text_upper))
            
            # Weighted scoring logic
            score = (keyword_match_count * 0.1) + (pattern_match_count * 0.5)
            scores[doc_type] = score * config["weight"]

        if not scores or max(scores.values()) == 0:
            return {"document_type": "NONE", "confidence": 0.0, "is_sensitive_document": False}

        best_type = max(scores, key=scores.get)
        # Normalize score to a pseudo-confidence [0, 1]
        best_score = scores[best_type]
        confidence = min(1.0, best_score / 2.0) 

        is_sensitive = confidence >= SENSITIVITY_THRESHOLD

        return {
            "document_type": best_type,
            "confidence": round(confidence, 2),
            "is_sensitive_document": is_sensitive
        }

# Singleton instance for the module
_classifier_instance = None

def classify_document(ocr_text: str) -> Dict:
    """Helper function for external modules to call the classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DocumentClassifier()
    return _classifier_instance.classify(ocr_text)

def get_masking_rules(document_type: str) -> Dict:
    """Returns masking rules for a specific document type."""
    return DOCUMENT_MASKING_RULES.get(document_type, DOCUMENT_MASKING_RULES["NONE"])
