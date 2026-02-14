import cv2
from ultralytics import YOLO
from .utils import apply_blackout

class ObjectDetector:
    """
    Detects sensitive objects (ID cards, credit cards, license plates) using YOLOv8.
    Applies black rectangle masking to detected objects.
    """
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.25):
        # Note: In a real scenario, you would use a model specialized for PII objects.
        # Here we use the standard YOLOv8n and define categories we consider sensitive.
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}. Object detection might be skipped.")
            self.model = None
            
        self.conf = confidence_threshold
        # Mapping for standard COCO model or potential custom model
        # Standard COCO doesn't have "ID Card", but we can define placeholders
        # In practice, you'd use a custom-trained model for these.
        self.sensitive_categories = ['license plate', 'id card', 'credit card', 'passport']

    def detect_and_mask(self, image):
        """
        Detects objects and masks them if they match sensitive categories.
        Returns the modified image and a list of detected sensitive objects.
        """
        detected_objects = []
        if self.model is None:
            return image, detected_objects

        results = self.model(image, conf=self.conf, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                
                # Check if the detected object is in our sensitive list
                # (Matches are often based on custom model labels)
                if label.lower() in [c.lower() for c in self.sensitive_categories]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    image = apply_blackout(image, (x1, y1, w, h))
                    detected_objects.append({
                        "label": label,
                        "confidence": float(box.conf[0]),
                        "box": (x1, y1, w, h)
                    })
                    
        return image, detected_objects
