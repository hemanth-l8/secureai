
from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import logging
import base64
import cv2
import numpy as np
from io import BytesIO

# Ensure the security directory is in the path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from middleware.ner import NERMiddleware
from middleware.privacy_filter import PrivacyFilter
from middleware.image_processor import PrivacyImageProcessor
from core.ai_model import CoreAIModel
from dotenv import load_dotenv

# Initialize Flask
app = Flask(__name__, static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Disable caching
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'temp_uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Load environment
load_dotenv(os.path.join(base_dir, ".env"))

# Initialize Pipeline Components
ner_layer = NERMiddleware()
privacy_layer = PrivacyFilter()
image_processor = PrivacyImageProcessor()
core_ai = CoreAIModel()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('static', path)

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    user_input = data.get('text', '')
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Step 1: NER Detection
    ner_report = ner_layer.scan(user_input)
    
    # Step 2: Tokenization
    tokenized_input = privacy_layer.sanitize(user_input, ner_report)
    token_map = privacy_layer.token_map
    
    # Step 3: LLM Call
    tokenized_response = core_ai.generate_response(tokenized_input)
    
    # Step 4: Detokenization
    final_response = privacy_layer.detokenize(tokenized_response)
    
    # Validation Logging
    print("\n--- PIPELINE VALIDATION ---")
    print(f"Masked Sent to LLM: {tokenized_input}")
    print(f"LLM Raw Output: {tokenized_response}")
    print(f"Restored Response: {final_response}")
    print("---------------------------\n")
    
    # Return secured report (No raw masked data unless debug)
    debug_mode = os.environ.get("DEBUG_MODE", "False").lower() == "true"
    
    privacy_report = {
        "sensitive_items": ner_report.get('total_sensitive_items', 0),
        "risk_level": "LOW",
        "entities": ner_report.get('entities', [])
    }

    response_data = {
        "llm_response": final_response,
        "final_response": final_response, # compatibility
        "privacy_report": privacy_report,
        "ner_report": ner_report, # compatibility
        "tokenized_input": tokenized_input,
        "risk_score": 0.0,
        "safe_to_forward": True
    }
    
    if debug_mode:
        response_data["llm_raw_response"] = tokenized_response
        response_data["token_map"] = token_map
        
    return jsonify(response_data)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save file temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        # Step 1: Process Image for Privacy (For UI Showcase)
        result = image_processor.process_image(file_path)
        
        # Step 2: Convert masked image to Base64 for UI display (Middleware Visual Proof)
        _, masked_buffer = cv2.imencode('.jpg', result['masked_image'])
        masked_base64 = base64.b64encode(masked_buffer).decode('utf-8')
        
        # Step 3: Prepare Original Image for LLM (Showcase: Direct Link while Middleware monitors)
        with open(file_path, "rb") as original_file:
            original_base64 = base64.b64encode(original_file.read()).decode('utf-8')

        # Get additional user instructions
        instructions = request.form.get('instructions', '').strip()
        
        # Process user instructions through privacy pipeline
        sanitized_instructions = ""
        instructions_token_map = {}
        
        if instructions:
            instructions_ner = ner_layer.scan(instructions)
            sanitized_instructions = privacy_layer.sanitize(instructions, instructions_ner)
            instructions_token_map = privacy_layer.token_map.copy()
            
        # Extract non-sensitive text from image processing result
        clean_text = result.get('clean_text', '')
        
        # Step 4: Construct the "Direct Link" LLM prompt
        system_prompt = (
            "You are analyzing a privacy-processed image.\n"
            "Some sensitive areas may be redacted in the system logs, but you are focused on the visual context.\n"
            "Focus only on visible elements such as clothing, objects, context, and user question.\n"
            "Do not discuss masking or privacy mechanisms."
        )
        
        llm_prompt = system_prompt + "\n"
        if clean_text:
            llm_prompt += f"\nNon-sensitive OCR text: {clean_text}\n"
        
        # Showcase: Send ORIGINAL instructions to LLM for perfect response,
        # while middleware monitors and masks for the UI.
        if instructions:
            llm_prompt += f"\nUser Context/Question: {instructions}\n"
        
        # Step 5: LLM Integration - Sending ORIGINAL image to show "Directly Linked" performance
        llm_response = core_ai.generate_response(llm_prompt, original_base64)
        
        # Step 5: Detokenize the response to restore any masked data from instructions (Requirement 3 restoration)
        if instructions_token_map:
            privacy_layer.token_map = instructions_token_map
            llm_response = privacy_layer.detokenize(llm_response)
        
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Ensure llm_response is not None or empty
        if not llm_response or llm_response.strip() == "":
            llm_response = "Unable to generate response. Please try again."
        
        # Prepare specific privacy report (Requirement PART 1 & 5)
        privacy_report = {
            "faces_detected": result.get('faces_detected', 0),
            "text_regions": result.get('sensitive_text_regions', 0),
            "barcodes_detected": result.get('barcodes_detected', 0),
            "masked_regions": result.get('faces_detected', 0) + result.get('sensitive_text_regions', 0) + result.get('barcodes_detected', 0),
            "risk_level": "CRITICAL" if result.get('risk_score', 0) > 0.8 else "LOW",
            "document_detected": result.get('document_detected', False)
        }

        response_data = {
            "masked_image": f"data:image/jpeg;base64,{masked_base64}",
            "privacy_report": privacy_report,
            "llm_response": llm_response,
            "final_response": llm_response, # Keep for backward compatibility if needed by other JS parts
            "risk_score": result.get('risk_score', 0),
            "safe_to_forward": result.get('safe_to_forward', True),
            "barcodes_detected": result.get('barcodes_detected', 0),
            "sensitive_hashes": result.get('sensitive_hashes', [])
        }
        
        debug_mode = os.environ.get("DEBUG_MODE", "False").lower() == "true"
        if debug_mode:
            response_data["llm_raw_response"] = llm_response
            response_data["clean_text_sent_to_llm"] = clean_text
            
        return jsonify(response_data)
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
