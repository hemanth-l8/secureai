import cv2
import os
import sys
from image_privacy import PrivacyPipeline

def run_demo(image_path):
    # Initialize pipeline with custom thresholds if desired
    config = {
        'face_conf': 0.5,
        'obj_conf': 0.25,
        'risk_threshold': 1.0,
        'languages': ['en']
    }
    
    pipeline = PrivacyPipeline(config)
    
    print(f"--- Processing Image: {image_path} ---")
    
    try:
        result = pipeline.process_image(image_path)
        
        print("\nResults:")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Faces Detected: {result['detected_faces']}")
        print(f"Objects Detected: {len(result['detected_objects'])}")
        print(f"Sensitive Text Found: {len(result['detected_sensitive_text'])}")
        print(f"Safe to Forward: {result['safe_to_forward']}")
        print(f"Categories Flags: {result['detected_categories']}")
        
        # Save the masked image
        output_path = "masked_output.jpg"
        cv2.imwrite(output_path, result['masked_image'])
        print(f"\nMasked image saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Instruction for dependencies
    print("Ensure you have installed: pip install mediapipe ultralytics easyocr opencv-python numpy")
    
    # Check if a path was provided
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Placeholder for user's own image
        img_path = "sample_image.jpg"
        if not os.path.exists(img_path):
            print(f"Please provide an image path or place a '{img_path}' in the directory.")
            sys.exit(1)
            
    run_demo(img_path)
