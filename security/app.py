
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
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'temp_uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    
    # Return full transparent report
    return jsonify({
        "original_input": user_input,
        "ner_report": ner_report,
        "tokenized_input": tokenized_input,
        "token_map": token_map,
        "llm_raw_response": tokenized_response,
        "final_response": final_response
    })

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
        # Step 1: Process Image for Privacy
        result = image_processor.process_image(file_path)
        
        # Step 2: Convert masked image to Base64 for UI
        _, buffer = cv2.imencode('.jpg', result['masked_image'])
        masked_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Step 3: Send context to LLM
        # We send a description of what was found (without sensitive details)
        llm_prompt = f"The user uploaded an image. Detected faces: {result['faces_detected']}. " \
                     f"Sensitive text regions masked: {result['sensitive_text_regions']}. " \
                     f"Is it an official document? {result['document_detected']}. " \
                     f"Risk Score: {result['risk_score']}. " \
                     f"Please provide a safety summary and advice for the user regarding this upload."
        
        llm_response = core_ai.generate_response(llm_prompt)
        
        # Cleanup
        os.remove(file_path)
        
        return jsonify({
            "masked_image": f"data:image/jpeg;base64,{masked_base64}",
            "risk_score": result['risk_score'],
            "faces_detected": result['faces_detected'],
            "sensitive_text_regions": result['sensitive_text_regions'],
            "document_detected": result['document_detected'],
            "safe_to_forward": result['safe_to_forward'],
            "llm_response": llm_response
        })
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
