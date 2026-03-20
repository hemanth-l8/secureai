
# Centralized Regex Patterns for Structured Sensitive Data
# Based on existing securea configuration

PATTERNS = {
    "AADHAAR": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "PHONE": r"\b[6-9]\d{9}\b", 
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "BANK_ACCOUNT": r"\b\d{9,18}\b",
    "REGISTRATION_ID": r"\b[0-9A-Z]{8,12}\b", # Catches roll numbers like 23PA1A05M2
    "OFFICIAL_ID": r"\b(?:ID|ROLL|REG|NO|NUM)\s*[:.-]?\s*[A-Z0-9-]{5,}\b"
}

# Contextual patterns that require a specific prefix or format
CONTEXTUAL_PATTERNS = {
    "USERNAME": r"(?i)(?:send msg to|message|contact|dm)\s+([A-Za-z0-9._]+)",
    "HANDLE": r"(@[A-Za-z0-9._]+)",
    "NAME": r"(?i)(?:this is|i am|i'm|my name is)\s+([A-Za-z\s]+)",
    "ENTITY_NAME": r"\b[A-Z][A-Z\s]{5,}\b" # Blocks of CAPS often used for names in ID cards
}


