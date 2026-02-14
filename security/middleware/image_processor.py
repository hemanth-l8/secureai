import cv2
import numpy as np
import easyocr
import mediapipe as mp
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrivacyImageProcessor:
    """
    Advanced Image Privacy Protection Pipeline.
    Detects and masks faces and sensitive text within documents/images.
    """
    def __init__(self, languages=['en']):
        # Initialize OCR Engine
        try:
            self.reader = easyocr.Reader(languages, gpu=False)
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None

        # Initialize Face Detector (MediaPipe with OpenCV fallback)
        self.face_detector = None
        self.face_fallback = False
        try:
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    min_detection_confidence=0.5
                )
            else:
                self.face_fallback = True
        except Exception:
            self.face_fallback = True
        
        if self.face_fallback:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detection: Using OpenCV Haar Cascade fallback.")
        else:
            logger.info("Face detection: Using MediaPipe.")

        # Sensitive Keywords & Document Identifiers
        self.doc_keywords = ["NATIONAL IDENTIFICATION", "PASSPORT", "DRIVER LICENSE", "ID CARD", "IDENTITY"]
        self.sensitive_keywords = ["NAME", "ID", "TEL", "DOB", "EXP", "ADDRESS", "GENDER", "ISSUE", "BLOOD"]
        
        # Regex Patterns
        self.patterns = {
            "name": r'\b[A-Z][A-Z\s]{5,}\b', # Simplified: Look for blocks of CAPS in documents
            "id_num": r'\b(?:ID#?|NUMBER|NO)\s*[:.]?\s*[A-Z0-9-]{6,}\b',
            "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "date": r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
            "alphanumeric_id": r'\b[A-Z0-9]{8,15}\b'
        }

    def detect_faces(self, image):
        """Detects faces and returns a list of bounding boxes [x, y, w, h]."""
        face_boxes = []
        if not self.face_fallback:
            results = self.face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.detections:
                h, w, _ = image.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(w - x, int(bbox.width * w))
                    height = min(h - y, int(bbox.height * h))
                    face_boxes.append((x, y, width, height))
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                face_boxes.append((x, y, w, h))
        return face_boxes

    def extract_text(self, image):
        """Extracts text and boxes using EasyOCR."""
        if self.reader is None:
            return []
        
        results = self.reader.readtext(image)
        extracted = []
        for (bbox, text, prob) in results:
            # Convert 4-point polygon to x, y, w, h
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x, y = int(min(x_coords)), int(min(y_coords))
            w, h = int(max(x_coords) - x), int(max(y_coords) - y)
            
            extracted.append({
                "text": text,
                "confidence": prob,
                "box": (x, y, w, h)
            })
        return extracted

    def detect_sensitive_text(self, ocr_results):
        """Analyzes text for sensitive patterns and keywords. Returns boxes to mask."""
        mask_boxes = []
        is_document = False
        
        # Check for Document Identification
        full_text = " ".join([item['text'].upper() for item in ocr_results])
        if any(keyword in full_text for keyword in self.doc_keywords):
            is_document = True
            logger.info("Document detected. Auto-masking policy triggered.")

        for item in ocr_results:
            text = item['text']
            text_upper = text.upper()
            box = item['box']
            should_mask = False
            
            # Policy 1: If it's a known sensitive document, mask ALL text fields
            if is_document:
                should_mask = True
            else:
                # Policy 2: Check for keywords
                if any(kw in text_upper for kw in self.sensitive_keywords):
                    should_mask = True
                
                # Policy 3: Check Regex Patterns
                if not should_mask:
                    for label, pattern in self.patterns.items():
                        if re.search(pattern, text, re.IGNORECASE):
                            should_mask = True
                            break
            
            if should_mask:
                mask_boxes.append(box)
                
        return mask_boxes, is_document

    def mask_regions(self, image, face_boxes, text_boxes):
        """Applies Gaussian blur to faces and black rectangles to sensitive text."""
        # 1. Mask Faces (Blur)
        for (x, y, w, h) in face_boxes:
            roi = image[y:y+h, x:x+w]
            # Ensure ROI is valid
            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, (51, 51), 30)
                image[y:y+h, x:x+w] = blurred

        # 2. Mask Text (Black Rectangle/Pixelation)
        for (x, y, w, h) in text_boxes:
            # Applying Blackout as it's more secure for text
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
            
        return image

    def calculate_risk(self, face_count, text_count, doc_detected):
        """
        logic: (faces * 0.2) + (text_regions * 0.5) + (doc_detected * 0.3)
        Threshold: 0.8
        """
        doc_val = 1 if doc_detected else 0
        risk_score = (face_count * 0.2) + (text_count * 0.5) + (doc_val * 0.3)
        risk_score = min(2.0, round(risk_score, 2)) # Cap at 2.0 for reporting
        
        safe = risk_score <= 0.8
        return risk_score, safe

    def process_image(self, image_path: str):
        """Main pipeline orchestration."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Incompatible image format or corrupted file.")

            # 1. Detection
            logger.info("Detecting faces...")
            face_boxes = self.detect_faces(image)
            
            logger.info("Extracting OCR text...")
            ocr_results = self.extract_text(image)
            
            logger.info("Analyzing sensitivity...")
            text_mask_boxes, doc_detected = self.detect_sensitive_text(ocr_results)

            # 2. Risk Evaluation
            risk_score, safe_to_forward = self.calculate_risk(
                len(face_boxes), 
                len(text_mask_boxes), 
                doc_detected
            )

            # 3. Masking (Work on a copy to keep original if needed)
            masked_image = self.mask_regions(image.copy(), face_boxes, text_mask_boxes)

            return {
                "masked_image": masked_image,
                "risk_score": risk_score,
                "faces_detected": len(face_boxes),
                "sensitive_text_regions": len(text_mask_boxes),
                "document_detected": doc_detected,
                "safe_to_forward": safe_to_forward
            }

        except Exception as e:
            logger.error(f"Error in processing: {e}")
            raise e

if __name__ == "__main__":
    import sys
    processor = PrivacyImageProcessor()
    
    path = "assets/test.jpg.png" if len(sys.argv) < 2 else sys.argv[1]
    
    try:
        result = processor.process_image(path)
        print("\n--- Advanced Privacy Pipeline Results ---")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Safe to Forward: {result['safe_to_forward']}")
        print(f"Faces: {result['faces_detected']}")
        print(f"Sensitive Text Boxes: {result['sensitive_text_regions']}")
        print(f"Official Document Detected: {result['document_detected']}")
        
        cv2.imwrite("advanced_masked_output.jpg", result['masked_image'])
        print("\nMasked output saved to: advanced_masked_output.jpg")
    except Exception as e:
        print(f"Execution failed: {e}")
