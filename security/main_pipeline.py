
import logging
import sys
import os

from dotenv import load_dotenv

# Setup paths (ensure utils/patterns visible)
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from middleware.ner import NERMiddleware
from middleware.privacy_filter import PrivacyFilter
from core.ai_model import CoreAIModel

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Environment explicitly
load_dotenv(os.path.join(base_dir, ".env"))

class SecureAIPipeline:
    """
    Orchestrates the secure pipeline:
    User Input -> NER Middleware -> Privacy Filter -> Core AI Model -> Response
    """
    
    def __init__(self):
        # Initialize sub-modules
        self.ner_layer = NERMiddleware()
        self.privacy_layer = PrivacyFilter()
        self.core_ai = CoreAIModel()
        
    def process_request(self, user_input):
        print("-" * 60)
        logging.info("PIPELINE: Received User Input.")
        
        # Step 1: NER Detection (Middleware)
        ner_report = self.ner_layer.scan(user_input)
        
        # Step 2: Risk Assessment (Logging Only - NO BLOCKING)
        total_sensitive = ner_report['total_sensitive_items']
        logging.info(f"PIPELINE: NER Report: {ner_report}")
        logging.info(f"PIPELINE: Detected {total_sensitive} sensitive items. Proceeding with Tokenization.")
        
        # Step 3: Tokenization (Middleware)
        # Sanitized input now contains tokens like [PERSON_1], [EMAIL_2]
        tokenized_input = self.privacy_layer.sanitize(user_input, ner_report)
        logging.info(f"PIPELINE: Tokenized Prompt sent to LLM: {tokenized_input}")
        
        # Step 4: Core AI Model Execution (Untouched)
        # The AI sees: "Review request for [PERSON_1]..."
        logging.info("PIPELINE: Forwarding to Core AI Model...")
        tokenized_response = self.core_ai.generate_response(tokenized_input)
        logging.info(f"PIPELINE: Raw LLM Response: {tokenized_response}")
        
        # Step 5: Detokenization (Middleware)
        # Restore original values so the user gets a seamless answer
        final_response = self.privacy_layer.detokenize(tokenized_response)
        
        return final_response

# --- INTERACTIVE TERMINAL APP ---

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SECURE MIDDLEWARE PIPELINE (NER + Privacy Filter)")
    print("Type 'exit' to quit.")
    print("="*60 + "\n")
    
    pipeline = SecureAIPipeline()
    
    while True:
        try:
            user_input = input("Enter prompt > ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
                
            if not user_input.strip():
                continue
                
            result = pipeline.process_request(user_input)
            print(f"\n>> AI RESPONSE: {result}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(f"Pipeline Error: {e}")
