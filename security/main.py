import re
import hashlib
import os
import sys
import time
import logging
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from dotenv import load_dotenv

# Load environment variables
def load_env():
    # Explicitly load .env from the same directory as this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(BASE_DIR, ".env")
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.info(f"Loaded .env from: {env_path}")
    else:
        logging.warning(f".env file not found at: {env_path}")

    # Verify API Key Loading (Masked)
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        logging.info(f"API Key loaded successfully: {api_key[:4]}****")
    else:
        logging.error("LLM_API_KEY not found in environment variables!")

load_env()

# --- CONFIGURATION & PATTERNS ---

class RiskLevel:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

PATTERNS = {
    "AADHAAR": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "PHONE": r"\b[6-9]\d{9}\b", # Assuming standard 10-digit Indian mobile numbers for simplicity/accuracy
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "BANK_ACCOUNT": r"\b\d{9,18}\b"
}

# --- CORE FUNCTIONS ---

def detect_sensitive_data(text):
    """
    Scans the text for sensitive patterns.
    Returns a dictionary of detected entities and their counts.
    """
    detected = {}
    total_count = 0
    
    for label, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            detected[label] = matches
            total_count += len(matches)
            
    return detected, total_count

def hash_data(data):
    """SHA-256 Hashing"""
    return hashlib.sha256(data.encode()).hexdigest()

def mask_phone(phone):
    """Mask phone number: XXXXXX1234"""
    clean_phone = re.sub(r'\D', '', phone) # Remove non-digits
    if len(clean_phone) >= 4:
        return 'X' * (len(clean_phone) - 4) + clean_phone[-4:]
    return 'X' * len(phone)

def mask_email(email):
    """Mask email: u***@domain.com"""
    if '@' in email:
        user, domain = email.split('@', 1)
        if len(user) > 1:
            masked_user = user[0] + '*' * (len(user) - 1)
        else:
            masked_user = '*'
        return f"{masked_user}@{domain}"
    return email

def morph_sensitive_data(text):
    """
    Replaces sensitive data with hashed or masked versions.
    """
    sanitized_text = text
    
    # We need to process patterns carefully to avoid overlapping replacements destroying data
    # Strategy: Find all matches first, then replace.
    # To avoid issues, we can iterate by type.
    
    # 1. Hashing (Aadhaar, PAN, Credit Card)
    for label in ["AADHAAR", "PAN", "CREDIT_CARD"]:
        matches = re.findall(PATTERNS[label], sanitized_text)
        for match in matches:
            hashed_value = f"[HASH:{hash_data(match)[:10]}...]" # Truncated hash for readability
            sanitized_text = sanitized_text.replace(match, hashed_value)

    # 2. Masking (Phone, Email, Bank Account)
    matches = re.findall(PATTERNS["PHONE"], sanitized_text)
    for match in matches:
        sanitized_text = sanitized_text.replace(match, mask_phone(match))
        
    matches = re.findall(PATTERNS["EMAIL"], sanitized_text)
    for match in matches:
        sanitized_text = sanitized_text.replace(match, mask_email(match))
        
    # Bank Account - treating like Credit Card for morphing (Hash) or Phone (Mask).
    # Requirement didn't specify Bank Account morphing, but let's default to Hashing as it's sensitive numeric.
    # Actually, let's mask it similar to Phone for usability or Hash it. Let's Hash it for security.
    matches = re.findall(PATTERNS["BANK_ACCOUNT"], sanitized_text)
    for match in matches:
         # Need to be careful not to double replace if overlaps (e.g. credit card within bank account pattern?)
         # Patterns are distinct enough usually. CC is 16, Bank 9-18.
         # If existing text was already replaced by CC hash, it won't match digits.
         if match in text: # check if it's still there (approximate)
             hashed_value = f"[HASH:{hash_data(match)[:10]}...]"
             sanitized_text = sanitized_text.replace(match, hashed_value)

    # 3. Token Replacement (IP Address)
    matches = re.findall(PATTERNS["IP_ADDRESS"], sanitized_text)
    for match in matches:
        sanitized_text = sanitized_text.replace(match, "[IP_ADDRESS]")
        
    return sanitized_text


def get_api_key():
    """Securely retrieves the API key from environment variables."""
    return os.environ.get("LLM_API_KEY")

def call_llm_api(sanitized_text):
    """
    Sends the sanitized prompt to the LLM API (gpt-4o-mini).
    """
    api_key = get_api_key()
    if not api_key:
        logging.warning("LLM_API_KEY not found in environment variables.")
        return "Error: API Key missing."
    
    logging.info("Connecting to LLM API...")
    
    # Auto-detect provider based on Key Prefix
    if api_key.startswith("gsk_"):
        provider = "Groq"
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        model = "llama-3.3-70b-versatile" # Updated valid model
    else:
        provider = "OpenAI"
        api_url = "https://api.openai.com/v1/chat/completions"
        model = "gpt-4o-mini"
        
    logging.info(f"Detected Provider: {provider} | Model: {model}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a secure AI assistant operating behind a privacy firewall. \n      Sensitive information has already been sanitized before reaching you.\n      Provide helpful, safe, and professional responses."
            },
            {
                "role": "user",
                "content": sanitized_text
            }
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        elif response.status_code == 401:
            logging.error("API Error 401: Unauthorized. Invalid API Key.")
            return "Error: Unauthorized. Check your API Key."
        elif response.status_code == 429:
            logging.error("API Error 429: Rate limit exceeded.")
            return "Error: Rate limit exceeded. Try again later."
        else:
            logging.error(f"API Error {response.status_code}: {response.text}")
            return f"Error: API Request failed with status {response.status_code}."
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Network Connection Error: {e}")
        return "Error: Failed to connect to LLM API."

def calculate_risk_level(total_sensitive_count):
    if total_sensitive_count == 0:
        return RiskLevel.LOW
    elif 1 <= total_sensitive_count <= 2:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.HIGH

def process_user_input(text):
    """
    Main orchestration function.
    """
    print("-" * 60)
    logging.info("Processing new request...")
    logging.info(f"Original Input: {text}")
    
    # 1. Detect
    detected_entities, count = detect_sensitive_data(text)
    risk_level = calculate_risk_level(count)
    
    # Log Detection
    if count > 0:
        logging.info(f"Detected Sensitive Entities: {detected_entities}")
        logging.info(f"Total Sensitive Entities Detected: {count}")
    else:
        logging.info("No sensitive data detected.")
        
    logging.info(f"Risk Level: {risk_level}")
    
    # 2. DECISION: Block if HIGH risk
    if risk_level == RiskLevel.HIGH:
        logging.error("BLOCKING REQUEST: High sensitive data exposure detected.")
        return "Request blocked due to high sensitive data exposure."
    
    # 3. Morph/Sanitize
    sanitized_text = morph_sensitive_data(text)
    logging.info(f"Sanitized Prompt: {sanitized_text}")
    
    # 4. Call LLM
    response = call_llm_api(sanitized_text)
    return response

# --- INTERACTIVE LOOP ---

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SECURE LLM GATEWAY (Backend Dummy Model)")
    print("Type 'exit' to quit.")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("Enter prompt > ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
                
            if not user_input.strip():
                continue
                
            result = process_user_input(user_input)
            print(f"\n>> SYSTEM OUTPUT: {result}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(f"An error occurred: {e}")
