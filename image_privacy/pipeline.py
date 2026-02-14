import cv2
import os
import logging
from .face_detector import FaceDetector
from .object_detector import ObjectDetector
from .ocr_engine import OCREngine
from .sensitive_text_analyzer import SensitiveTextAnalyzer
from .risk_engine import RiskEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrivacyPipeline:
    """
    Modular Multimodal Image Processing Pipeline for Privacy.
    Orchestrates face detection, object detection, OCR, and risk assessment.
    """
    def __init__(self, config=None):
        if config is None:
            config = {}
            
        self.face_detector = FaceDetector(
            min_detection_confidence=config.get('face_conf', 0.5)
        )
        self.object_detector = ObjectDetector(
            confidence_threshold=config.get('obj_conf', 0.25)
        )
        self.ocr_engine = OCREngine(
            languages=config.get('languages', ['en'])
        )
        self.text_analyzer = SensitiveTextAnalyzer()
        self.risk_engine = RiskEngine(
            threshold=config.get('risk_threshold', 1.0)
        )

    def process_image(self, image_path: str) -> dict:
        """
        Main entry point to process an image through the privacy pipeline.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image at: {image_path}")

            # 1. Face Detection & Masking (Blur)
            logger.info("Detecting faces...")
            image, face_count = self.face_detector.detect_and_mask(image)

            # 2. Object Detection & Masking (Blackout)
            logger.info("Detecting sensitive objects...")
            image, detected_objects = self.object_detector.detect_and_mask(image)

            # 3. OCR Extraction
            logger.info("Extracting text...")
            ocr_results = self.ocr_engine.extract_text_with_boxes(image)

            # 4. Sensitive Text Analysis & Masking (Pixelation)
            logger.info("Analyzing text for sensitive patterns...")
            image, detected_sensitive_text = self.text_analyzer.analyze_and_mask(image, ocr_results)

            # 5. Risk Assessment
            logger.info("Computing risk score...")
            risk_score, safe_to_forward = self.risk_engine.compute_risk(
                face_count=face_count,
                object_count=len(detected_objects),
                text_count=len(detected_sensitive_text)
            )

            # Prepare categories list for summary
            detected_categories = ["faces"] if face_count > 0 else []
            detected_categories.extend([obj['label'] for obj in detected_objects])
            for st in detected_sensitive_text:
                detected_categories.extend(st['types'])
            
            # De-duplicate categories
            detected_categories = list(set(detected_categories))

            return {
                "masked_image": image,
                "risk_score": risk_score,
                "detected_faces": face_count,
                "detected_objects": detected_objects,
                "detected_sensitive_text": detected_sensitive_text,
                "detected_categories": detected_categories,
                "safe_to_forward": safe_to_forward
            }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise e
