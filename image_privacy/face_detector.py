import cv2
import logging
from .utils import apply_gaussian_blur

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Detects human faces. Uses MediaPipe if available, falls back to OpenCV Haar Cascades.
    """
    def __init__(self, min_detection_confidence=0.5):
        self.face_detection = None
        self.fallback = False
        
        try:
            import mediapipe as mp
            # Check if mediapipe solutions are present
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                self.face_detection = mp.solutions.face_detection.FaceDetection(
                    min_detection_confidence=min_detection_confidence
                )
                logger.info("Using MediaPipe for face detection.")
            else:
                # Attempt to use tasks API or just fallback
                logger.info("MediaPipe solutions not found. Falling back to OpenCV.")
                self.fallback = True
        except (ImportError, AttributeError):
            logger.info("MediaPipe not available. Falling back to OpenCV.")
            self.fallback = True

        if self.fallback:
            # Load OpenCV's built-in Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detection = cv2.CascadeClassifier(cascade_path)

    def detect_and_mask(self, image):
        """
        Detects faces and applies Gaussian blur.
        """
        face_count = 0
        
        if not self.fallback:
            # MediaPipe Logic
            results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.detections:
                face_count = len(results.detections)
                h, w, _ = image.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(w - x, int(bbox.width * w))
                    height = min(h - y, int(bbox.height * h))
                    image = apply_gaussian_blur(image, (x, y, width, height))
        else:
            # OpenCV Fallback Logic
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detection.detectMultiScale(gray, 1.1, 4)
            face_count = len(faces)
            for (x, y, w, h) in faces:
                image = apply_gaussian_blur(image, (x, y, w, h))
                
        return image, face_count
