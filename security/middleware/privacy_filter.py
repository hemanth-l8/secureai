
import hashlib
import re
from utils.patterns import PATTERNS

class PrivacyFilter:
    """
    Applies reversible tokenization to detected sensitive data.
    """
    def __init__(self):
        self.token_map = {}
        self.counter = 0

    def _generate_token(self, label):
        self.counter += 1
        return f"[{label}_{self.counter}]"

    def sanitize(self, text, ner_report):
        """
        Replaces sensitive data with unique tokens and stores the mapping.
        Returns: (sanitized_text, token_map)
        """
        sanitized_text = text
        self.token_map = {} # Reset for new request
        self.counter = 0
        
        # Helper to process entities
        def process_entity(label, item):
            # Check if we already tokenized this exact item to maintain consistency
            existing_token = None
            for token, original in self.token_map.items():
                if original == item:
                    existing_token = token
                    break
            
            if existing_token:
                return existing_token
            
            # Create new token
            token = self._generate_token(label)
            self.token_map[token] = item
            return token

        # 1. Morph Structured Data (Regex Findings)
        structured_data = ner_report.get("structured_data", {})
        for label, items in structured_data.items():
            for item in items:
                # Replace with Token
                token = process_entity(label, item)
                sanitized_text = sanitized_text.replace(item, token)
        
        # 2. Morph Contextual Regex Data (Usernames)
        contextual_regex_data = ner_report.get("contextual_regex_data", {})
        for label, items in contextual_regex_data.items():
            for item in items:
                token = process_entity(label, item)
                sanitized_text = sanitized_text.replace(item, token)

        # 3. Morph Contextual Data (ML Findings)
        contextual_data = ner_report.get("contextual_data", {})
        for label, items in contextual_data.items():
            for item in items:
                token = process_entity(label, item)
                sanitized_text = sanitized_text.replace(item, token)
                
        return sanitized_text

    def detokenize(self, text):
        """
        Restores original data from tokens in the text.
        Handles both bracketed [TOKEN_N] and stripped TOKEN_N formats.
        """
        detokenized_text = text
        
        # Sort tokens by length (descending) to avoid partial replacement issues
        sorted_tokens = sorted(self.token_map.keys(), key=len, reverse=True)
        
        for token in sorted_tokens:
            original = self.token_map[token]
            # 1. Replace exact bracketed match
            detokenized_text = detokenized_text.replace(token, original)
            
            # 2. Replace stripped match (e.g., USERNAME_1) with word boundaries
            stripped_token = token.strip("[]")
            # Only replace if it's not already replaced (brackets gone)
            # Use regex for word boundaries to avoid replacing "USERNAME_10" when looking for "USERNAME_1"
            detokenized_text = re.sub(rf'\b{re.escape(stripped_token)}\b', original, detokenized_text)
            
        return detokenized_text

def get_filter_module():
    return PrivacyFilter()
