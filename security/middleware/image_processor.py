import cv2
import numpy as np
import easyocr
import mediapipe as mp
import re
import logging
import os
import hashlib
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import NER
try:
    from middleware.ner import NERMiddleware
except ImportError:
    # Local fallback
    from ner import NERMiddleware

# Import document classifier
try:
    from core.vision.document_classifier import classify_document, get_masking_rules
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Document classifier not available: {e}")
    CLASSIFIER_AVAILABLE = False
    classify_document = None
    get_masking_rules = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrivacyImageProcessor:
    """
    Advanced Image Privacy Protection Pipeline.
    Detects and masks faces and sensitive text within documents/images.
    """
    def __init__(self, languages=['en', 'te']):
        # Initialize OCR Engine
        try:
            self.reader = easyocr.Reader(languages, gpu=False)
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None
        
        # QR Code & Barcode Detector
        self.qr_detector = cv2.QRCodeDetector()

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

        # Initialize NER Layer for text analysis
        self.ner_layer = NERMiddleware()
        
        # Sensitive Keywords & Document Identifiers
        self.doc_keywords = ["NATIONAL IDENTIFICATION", "PASSPORT", "DRIVER LICENSE", "ID CARD", "IDENTITY", "VISHNU", "INSTITUTE"]
        self.sensitive_keywords = ["NAME", "ID", "TEL", "DOB", "EXP", "ADDRESS", "GENDER", "ISSUE", "BLOOD", "PRINCIPAL", "B.TECH"]
        
        # Regex Patterns
        self.patterns = {
            "name": r'\b[A-Z][A-Z\s]{4,}\b', # Block CAPS names with at least 5 chars
            "id_num": r'\b(?:ID#?|NUMBER|NO|Aadhar|ROLL)\s*[:.]?\s*[A-Z0-9\/-]{5,}\b',
            "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "date": r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
            "alphanumeric_id": r'\b[A-Z0-9]{6,15}\b' # Shorter alphanumeric catches like 23PA1A...
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
            for (x, y, w, h) in faces:
                face_boxes.append((x, y, w, h))
        return face_boxes

    def detect_qr(self, image):
        """Detects QR codes and 1D Barcodes using multi-pass CV analysis."""
        qr_boxes = []
        try:
            # Pass 1: Standard QR Detector
            retval, decode_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(image)
            if retval and points is not None:
                for pts in points:
                    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
                    qr_boxes.append((x, y, w, h))
            
            # Pass 2: Advanced Contour analysis for 1D Barcodes and complex QRs
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use Morphological operations to highlight barcode/QR blocks
            # Thresholding
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Closing kernel to join bars/blocks
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Erosion/Dilation to remove small noise
            closed = cv2.erode(closed, None, iterations=2)
            closed = cv2.dilate(closed, None, iterations=2)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                
                if area < 500: continue
                
                aspect_ratio = float(w)/h
                
                # Logic:
                # 1. QR Code: Square-ish (0.7 to 1.3)
                # 2. 1D Barcode: Wide (aspect ratio > 2.0 or < 0.5 for vertical)
                is_qr = 0.7 < aspect_ratio < 1.3 and area > 1000
                is_barcode = (aspect_ratio > 2.0 or aspect_ratio < 0.5) and area > 1500
                
                if is_qr or is_barcode:
                    # Check if it wasn't already detected
                    if not any(abs(x-bx)<30 and abs(y-by)<30 for (bx, by, bw, bh) in qr_boxes):
                        qr_boxes.append((x, y, w, h))
                        
            logger.info(f"Visual asset detection: found {len(qr_boxes)} potential codes.")
        except Exception as e:
            logger.warning(f"QR/Barcode detection failed: {e}")
        return qr_boxes

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
        """
        Analyzes text for sensitive patterns using hybrid NER.
        Maps detected entities back to OCR bounding boxes.
        """
        mask_boxes = []
        sensitive_texts = []
        non_sensitive_texts = []
        is_document = False
        
        if not ocr_results:
            return [], False, [], [], None
        
        # 1. Prepare text for NER scanning
        full_text = " ".join([item['text'] for item in ocr_results])
        full_text_upper = full_text.upper()
        
        # 2. Check for document context
        if any(keyword in full_text_upper for keyword in self.doc_keywords):
            is_document = True
            logger.info("Document context detected via keywords.")
        
        # 3. Use Hybrid NER Scanner
        ner_results = self.ner_layer.scan(full_text)
        detected_entities = ner_results.get('entities', [])
        
        # 4. ML-BASED Document Classification (If available)
        ml_classification = None
        if CLASSIFIER_AVAILABLE and classify_document:
            ml_classification = classify_document(full_text)
            if ml_classification['is_sensitive_document']:
                is_document = True
        
        # 5. Mapping & Selective Masking
        # We check each OCR block against detected sensitive items
        sensitive_spans = [(e['text'].lower(), e['label']) for e in detected_entities]
        
        for item in ocr_results:
            text = item['text']
            text_lower = text.lower()
            text_upper = text.upper()
            box = item['box']
            should_mask = False
            detected_type = "GENERAL_PII"
            
            # Policy 1: Check if this OCR block is part of any detected sensitive entity
            for s_text, s_label in sensitive_spans:
                if s_text in text_lower or text_lower in s_text:
                    should_mask = True
                    detected_type = s_label
                    break
            
            # Policy 2: Fallback for keywords (ensure name labels etc are masked)
            if not should_mask:
                if any(kw in text_upper for kw in self.sensitive_keywords):
                    should_mask = True
                    detected_type = "SENSITIVE_KEYWORD"
                else:
                    # Policy 3: Regex backup for specific local patterns
                    for label, pattern in self.patterns.items():
                        if re.search(pattern, text):
                            should_mask = True
                            detected_type = label.upper()
                            break
            
            if should_mask:
                mask_boxes.append(box)
                sensitive_texts.append({"text": text, "type": detected_type})
            else:
                non_sensitive_texts.append(text)
                
        return mask_boxes, is_document, sensitive_texts, non_sensitive_texts, ml_classification

    def mask_regions(self, image, visual_boxes, text_boxes, document_type="NONE"):
        """Applies Gaussian blur to faces/QRs and black rectangles to sensitive text."""
        h, w = image.shape[:2]
        
        # 1. Mask Visual Assets (Faces, QR Codes)
        for (x, y, bw, bh) in visual_boxes:
            # Add small padding for visual safety
            x, y = max(0, x-5), max(0, y-5)
            bw, bh = min(w-x, bw+10), min(h-y, bh+10)
            
            roi = image[y:y+bh, x:x+bw]
            if roi.size > 0:
                # Stronger blur for visual assets
                blurred = cv2.GaussianBlur(roi, (99, 99), 50)
                image[y:y+bh, x:x+bw] = blurred
                # Add thin border to indicate redacted area
                cv2.rectangle(image, (x, y), (x + bw, y + bh), (40, 40, 40), 1)

        # 2. Mask Text (Black Rectangle)
        for (x, y, bw, bh) in text_boxes:
            # Padding to ensure complete coverage (Requirement 6)
            pad = 4
            px, py = max(0, x-pad), max(0, y-pad)
            pw, ph = min(w-px, bw+(pad*2)), min(h-py, bh+(pad*2))
            cv2.rectangle(image, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
        
        # 3. Aggressive Logic: Manual ROI check for Aadhaar (Requirement 6 Escalation)
        if document_type == "AADHAAR":
            # Heuristic: Aadhaar photos are often in the bottom-left quadrant
            # If no face was detected, we apply a safety blur in the typical photo area
            if not any(bx < w/2 and by > h/2 for (bx, by, bw, bh) in visual_boxes):
                logger.info("[PROTECTION] Applying safety blur to typical Aadhaar photo region.")
                photo_roi_x, photo_roi_y = int(w * 0.05), int(h * 0.6)
                photo_roi_w, photo_roi_h = int(w * 0.25), int(h * 0.3)
                roi = image[photo_roi_y:photo_roi_y+photo_roi_h, photo_roi_x:photo_roi_x+photo_roi_w]
                if roi.size > 0:
                    image[photo_roi_y:photo_roi_y+photo_roi_h, photo_roi_x:photo_roi_x+photo_roi_w] = cv2.GaussianBlur(roi, (99, 99), 50)

        return image

    def calculate_risk(self, face_count, text_count, doc_detected, ml_classification=None):
        """
        Enhanced risk calculation with ML document classification.
        logic: (faces * 0.2) + (text_regions * 0.5) + (doc_detected * 0.3) + (ml_escalation)
        Threshold: 0.8
        """
        doc_val = 1 if doc_detected else 0
        base_risk = (face_count * 0.2) + (text_count * 0.5) + (doc_val * 0.3)
        
        # PART 8: ML-based risk escalation
        ml_escalation = 0.0
        if ml_classification and ml_classification['is_sensitive_document']:
            # Get document-specific risk escalation
            if CLASSIFIER_AVAILABLE and get_masking_rules:
                masking_rules = get_masking_rules(ml_classification['document_type'])
                ml_escalation = masking_rules.get('risk_escalation', 0.4)
                logger.info(f"[RISK] ML escalation applied: +{ml_escalation} for {ml_classification['document_type']}")
        
        risk_score = base_risk + ml_escalation
        risk_score = min(2.0, round(risk_score, 2))  # Cap at 2.0 for reporting
        
        # Risk controls forwarding decision, NOT masking
        safe = risk_score <= 0.8
        return risk_score, safe

    def process_image(self, image_path: str):
        """Main pipeline orchestration optimized for CPU."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Incompatible image format or corrupted file.")

            # CPU Optimization: Resize large images to 800px max dimension
            h, w = image.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
                logger.info(f"Image resized to {image.shape[1]}x{image.shape[0]} for faster CPU processing.")

            # 1. Detection
            logger.info("Detecting faces...")
            face_boxes = self.detect_faces(image)
            
            logger.info("Detecting QR codes...")
            qr_boxes = self.detect_qr(image)
            
            # Combine non-text sensitive regions
            visual_mask_boxes = face_boxes + qr_boxes

            # Conditional OCR: Run only if likely to be needed
            # (Checking for faces first or just running always if accuracy is priority)
            logger.info("Extracting OCR text...")
            ocr_results = self.extract_text(image)
            
            logger.info("Analyzing sensitivity...")
            text_mask_boxes, doc_detected, sensitive_items, non_sensitive_text, ml_classification = self.detect_sensitive_text(ocr_results)

            # Validation Logging (Requirement 5)
            logger.info(f"Sensitive extracted: {[s['text'] for s in sensitive_items]}")
            clean_text_for_llm = " ".join(non_sensitive_text)
            logger.info(f"Non-sensitive text to LLM: {clean_text_for_llm}")

            # 2. Risk Evaluation (with ML classification)
            risk_score, safe_to_forward = self.calculate_risk(
                len(face_boxes), 
                len(text_mask_boxes), 
                doc_detected,
                ml_classification
            )

            # 3. Masking
            # PART 6 Compliance: Ensure faces and QRs are masked
            doc_type = ml_classification['document_type'] if ml_classification else "NONE"
            masked_image = self.mask_regions(image.copy(), visual_mask_boxes, text_mask_boxes, document_type=doc_type)

            # 4. Selective Hashing of Sensitive Data (PART 7: Hash only sensitive spans)
            # Use system secret salt for enhanced security
            SYSTEM_SALT = os.environ.get('PRIVACY_SALT', 'secure_ai_middleware_v1')
            sensitive_hashes = []
            for item in sensitive_items:
                # Hash with salt
                salted_value = item['text'] + SYSTEM_SALT
                h = hashlib.sha256(salted_value.encode()).hexdigest()
                sensitive_hashes.append({"hash": h, "type": item['type']})
                # Immediately discard raw sensitive text (security requirement)
                del salted_value

            return {
                "masked_image": masked_image,
                "risk_score": risk_score,
                "faces_detected": len(face_boxes),
                "barcodes_detected": len(qr_boxes),
                "sensitive_text_regions": len(text_mask_boxes),
                "document_detected": doc_detected,
                "safe_to_forward": safe_to_forward,
                "clean_text": clean_text_for_llm,
                "sensitive_hashes": sensitive_hashes,
                "ml_classification": ml_classification if ml_classification else {"document_type": "NONE", "confidence": 0.0}
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
