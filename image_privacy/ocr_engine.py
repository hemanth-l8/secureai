import easyocr
import numpy as np

class OCREngine:
    """
    Extracts text and its bounding boxes from an image using EasyOCR.
    """
    def __init__(self, languages=['en']):
        try:
            self.reader = easyocr.Reader(languages, gpu=False)
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR: {e}")
            self.reader = None

    def extract_text_with_boxes(self, image):
        """
        Extracts text from the image.
        Returns a list of dicts containing text, confidence, and bounding box.
        """
        if self.reader is None:
            return []

        # results is a list of [bbox, text, confidence]
        results = self.reader.readtext(image)
        
        extracted_data = []
        for (bbox, text, prob) in results:
            # bbox is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            x, y = int(min(x_coords)), int(min(y_coords))
            w, h = int(max(x_coords) - x), int(max(y_coords) - y)
            
            extracted_data.append({
                "text": text,
                "confidence": float(prob),
                "box": (x, y, w, h)
            })
            
        return extracted_data
