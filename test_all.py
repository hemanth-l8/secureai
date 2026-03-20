import os
import sys
import cv2
import numpy as np
import logging

# Ensure the security directory is in the path for internal imports
root_dir = os.path.dirname(os.path.abspath(__file__))
security_dir = os.path.join(root_dir, "security")
sys.path.append(security_dir)

from advanced_processor import PrivacyImageProcessor
from security.middleware.ner import NERMiddleware
from security.middleware.privacy_filter import PrivacyFilter

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemTest")

def test_image_processor():
    logger.info("--- Testing Image Processor ---")
    processor = PrivacyImageProcessor()
    
    # Create a dummy image with text
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(img, "DRIVER LICENSE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "NAME: JOHN DOE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "ID: 1234-5678-9012", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    test_path = "test_image_tmp.jpg"
    cv2.imwrite(test_path, img)
    
    try:
        result = processor.process_image(test_path)
        logger.info(f"Risk Score: {result['risk_score']}")
        logger.info(f"Safe to Forward: {result['safe_to_forward']}")
        logger.info(f"Document Detected: {result['document_detected']}")
        
        assert result['document_detected'] == True
        assert result['risk_score'] >= 1.0
        logger.info("Image Processor Test PASSED")
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)

def test_text_pipeline():
    logger.info("--- Testing Text Privacy Pipeline ---")
    ner = NERMiddleware()
    filter_layer = PrivacyFilter()
    
    # Use a 10-digit number that matches PATTERNS["PHONE"]
    text = "My name is John and my phone is 9876543210. I live in New York."
    
    # NER Scan
    report = ner.scan(text)
    logger.info(f"Entities Found: {report['total_sensitive_items']}")
    
    # Sanitize
    sanitized = filter_layer.sanitize(text, report)
    logger.info(f"Sanitized Text: {sanitized}")
    
    # Check if Name and Phone/GPE are masked (labels might vary like [NAME_1] or [GPE_3])
    assert "[" in sanitized and "]" in sanitized
    assert "John" not in sanitized
    assert "9876543210" not in sanitized
    
    # Restore
    restored = filter_layer.detokenize(f"Hello {sanitized}")
    logger.info(f"Restored Text: {restored}")
    
    assert "John" in restored
    assert "9876543210" in restored
    logger.info("Text Pipeline Test PASSED")

if __name__ == "__main__":
    logger.info("Starting System-Wide Integration Test...")
    try:
        test_text_pipeline()
        test_image_processor()
        logger.info("\nALL SYSTEMS INTEGRATED AND VERIFIED SUCCESSFULLY!")
    except Exception as e:
        logger.error(f"Test Failed: {e}")
        exit(1)
