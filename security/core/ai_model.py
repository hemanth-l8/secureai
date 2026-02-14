
import os
import requests
import logging

def get_api_key():
    """Securely retrieves the API key from environment variables."""
    # Assumes .env is loaded before calling this or available in os.environ
    return os.environ.get("LLM_API_KEY")

class CoreAIModel:
    """
    Represents the existing, untouched Core AI Model.
    This class simulates the 'black box' AI that takes a prompt and returns a response.
    """
    
    def generate_response(self, text):
        """
        The main inference method.
        Directly calls the external LLM API (OpenAI/Groq).
        """
        api_key = get_api_key()
        if not api_key:
            logging.error("CoreAIModel: API Key missing.")
            return "Error: API Key missing."
        
        logging.info("CoreAIModel: Connecting to LLM API...")
        
        # Auto-detect provider based on Key Prefix logic (preserved from original main.py)
        if api_key.startswith("gsk_"):
            api_url = "https://api.groq.com/openai/v1/chat/completions"
            model = "llama-3.3-70b-versatile"
        else:
            api_url = "https://api.openai.com/v1/chat/completions"
            model = "gpt-4o-mini"
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Using a standard system prompt for the core AI
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": text
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
                content = response.json()['choices'][0]['message']['content']
                return content
            else:
                logging.error(f"CoreAIModel Error {response.status_code}: {response.text}")
                return f"Error: AI Service failed with status {response.status_code}."
                
        except Exception as e:
            logging.error(f"CoreAIModel Connection Error: {e}")
            return "Error: Failed to connect to AI Model."
