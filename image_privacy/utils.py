import cv2
import numpy as np

def apply_gaussian_blur(image, box, kernel_size=(51, 51), sigma=30):
    """Applies Gaussian blur to a specific region (box) of the image."""
    x, y, w, h = box
    roi = image[y:y+h, x:x+w]
    blurred_roi = cv2.GaussianBlur(roi, kernel_size, sigma)
    image[y:y+h, x:x+w] = blurred_roi
    return image

def apply_pixelation(image, box, blocks=10):
    """Applies pixelation effect to a specific region (box) of the image."""
    x, y, w, h = box
    roi = image[y:y+h, x:x+w]
    
    # Get input size
    (h_roi, w_roi) = roi.shape[:2]
    
    # Resize input to "blocks x blocks"
    temp = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    
    # Initialize output image
    pixelated_roi = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
    
    image[y:y+h, x:x+w] = pixelated_roi
    return image

def apply_blackout(image, box):
    """Applies a black rectangle to a specific region (box) of the image."""
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return image
