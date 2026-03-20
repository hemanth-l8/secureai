
import os
import requests
import logging
import base64
import cv2
import numpy as np

def get_api_key():
    """Securely retrieves the API key from environment variables."""
    return os.environ.get("LLM_API_KEY")

def compress_image_b64(image_b64, max_width=512, quality=70):
    """
    Decodes a base64 image, resizes it to max_width keeping aspect ratio,
    re-encodes as low-quality JPEG, and returns the new base64 string.
    This is critical: free LLM vision models reject large base64 payloads.
    """
    try:
        # Strip data URI prefix if present
        raw = image_b64
        if "," in raw:
            raw = raw.split(",")[1]

        img_bytes = base64.b64decode(raw)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return raw  # Return original if decode fails

        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode(".jpg", img, encode_params)
        compressed_b64 = base64.b64encode(buffer).decode("utf-8")
        logging.info(f"Image compressed: original={len(raw)} chars -> compressed={len(compressed_b64)} chars")
        return compressed_b64
    except Exception as e:
        logging.warning(f"Image compression failed, using original: {e}")
        return raw.split(",")[1] if "," in raw else raw


class CoreAIModel:
    """
    Represents the Core AI Model with multimodal vision support.
    Uses a fallback list of free OpenRouter vision models.
    Images are automatically compressed before sending to stay within free-tier limits.
    """

    OPENROUTER_FREE_VISION_MODELS = [
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-12b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
    ]

    OPENROUTER_FREE_TEXT_MODELS = [
        "google/gemma-3-27b-it:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
    ]

    def generate_response(self, text, image_b64=None):
        """
        Main inference method. Supports text and optional image input.
        Images are compressed before sending to stay within free-tier size limits.
        """
        api_key = get_api_key()
        if not api_key:
            logging.error("CoreAIModel: API Key missing.")
            return "Error: API Key missing."

        # Compress image before sending (prevents 404/413 from large payloads)
        compressed_b64 = None
        if image_b64:
            compressed_b64 = compress_image_b64(image_b64, max_width=512, quality=72)

        if api_key.startswith("sk-or-"):
            return self._call_openrouter(api_key, text, compressed_b64)
        elif api_key.startswith("gsk_"):
            api_url = "https://api.groq.com/openai/v1/chat/completions"
            # Groq's llama-3.2 vision models are decommissioned. Use text fallback.
            model = "llama-3.3-70b-versatile" 
            return self._call_api(api_url, api_key, model, text, None)
        else:
            api_url = "https://api.openai.com/v1/chat/completions"
            model = "gpt-4o"
            return self._call_api(api_url, api_key, model, text, compressed_b64)

    def _call_openrouter(self, api_key, text, image_b64=None):
        """Try each free model in sequence until one succeeds."""
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        model_list = self.OPENROUTER_FREE_VISION_MODELS if image_b64 else self.OPENROUTER_FREE_TEXT_MODELS

        for model in model_list:
            logging.info(f"CoreAIModel: Trying OpenRouter model: {model}")
            result = self._call_api(api_url, api_key, model, text, image_b64)
            if not result.startswith("Error: AI Service failed"):
                return result
            logging.warning(f"CoreAIModel: Model {model} failed, trying next...")

        return "Error: All free vision models are currently unavailable. Please try again later."

    def _call_api(self, api_url, api_key, model, text, image_b64=None):
        """Make a single API call with the given model."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Required by OpenRouter for free tier
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "SecureAI Middleware"
        }

        # Build multimodal content
        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})

        if image_b64:
            # image_b64 is already a clean base64 string (no prefix)
            img_src = f"data:image/jpeg;base64,{image_b64}"
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_src,
                    "detail": "low"   # Use low detail to reduce token usage on free tier
                }
            })

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            print(f"LLM [{model}] Status: {response.status_code}")
            if response.status_code != 200:
                print(f"LLM [{model}] Error Body: {response.text[:600]}")

            if response.status_code == 200:
                res_json = response.json()
                content = res_json['choices'][0]['message']['content']
                return content
            else:
                return f"Error: AI Service failed with status {response.status_code}."

        except Exception as e:
            print(f"LLM [{model}] Connection Error: {e}")
            return "Error: Failed to connect to AI Model."
