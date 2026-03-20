
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
        Replaces sensitive data with unique tokens using character spans.
        Returns: sanitized_text
        """
        self.token_map = {} # Reset for new request
        self.counter = 0
        
        entities = ner_report.get("entities", [])
        
        # Sort entities by start position in REVERSE order
        # This allows us to replace text without invalidating the indices of subsequent spans
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        sanitized_text = text
        
        for ent in sorted_entities:
            start = ent['start']
            end = ent['end']
            label = ent['label']
            original_val = ent['text']
            
            # Check if we already tokenized this exact string content elsewhere in this request
            # (Optional: some prefer different tokens for different locations, 
            # but usually same string = same token for consistency)
            existing_token = None
            for token, mapped_val in self.token_map.items():
                if mapped_val == original_val:
                    existing_token = token
                    break
            
            if existing_token:
                token = existing_token
            else:
                self.counter += 1
                token = f"[{label}_{self.counter}]"
                self.token_map[token] = original_val
            
            # Replace using string slicing (Span-based)
            sanitized_text = sanitized_text[:start] + token + sanitized_text[end:]
                
        return sanitized_text

    def detokenize(self, text):
        """
        Restores original data from tokens in the text using exact matches.
        Usage: [LABEL_N] -> original_value
        """
        detokenized_text = text
        
        # Sort tokens by length (descending) to avoid partial replacement issues
        # (Though with bracketed tokens, this is less critical)
        sorted_tokens = sorted(self.token_map.keys(), key=len, reverse=True)
        
        for token in sorted_tokens:
            original = self.token_map[token]
            # Replace only the exact bracketed match
            detokenized_text = detokenized_text.replace(token, original)
            
        return detokenized_text

def get_filter_module():
    return PrivacyFilter()
