import cv2
import os
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path

class ObjectDetectionCropper:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the object detection and cropping system
        
        Args:
            model_path (str): Path to YOLO model file
            confidence_threshold (float): Minimum confidence for detection
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # COCO dataset class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def create_output_folders(self, base_path="detected_objects"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Add timestamp to make unique folder
        output_dir = f"{base_path}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True) # Create base directory
        
        print(f"Created output directory: {output_dir}")
        return output_dir
    
    def detect_and_crop_objects(self, image_path, output_base_dir):
        image = cv2.imread(image_path) # Load image
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        
        # Run inference
        results = self.model(image)
        
        # Get detections
        detections = results[0].boxes
        
        if detections is None or len(detections) == 0:
            print("No objects detected in the image")
            return
        
        # Dictionary to count objects of each class
        class_counts = {}
        
        # Process each detection
        for i, detection in enumerate(detections):
            # Get detection info
            confidence = float(detection.conf)
            class_id = int(detection.cls)
            
            # Skip if confidence is below threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Get class name
            class_name = self.class_names[class_id]
            
            # Count objects of this class
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            
            # Create class-specific folder
            class_folder = os.path.join(output_base_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            
            # Crop object from image
            cropped_object = image[y1:y2, x1:x2]
            
            # Generate filename
            input_filename = Path(image_path).stem
            crop_filename = f"{input_filename}_{class_name}_{class_counts[class_name]:03d}_conf{confidence:.2f}.jpg"
            crop_path = os.path.join(class_folder, crop_filename)
            
            # Save cropped image
            cv2.imwrite(crop_path, cropped_object)
            
            print(f"Saved: {crop_path} (confidence: {confidence:.2f})")
        
        # Create annotated image with bounding boxes
        annotated_image = results[0].plot()
        annotated_path = os.path.join(output_base_dir, f"annotated_{Path(image_path).name}")
        cv2.imwrite(annotated_path, annotated_image)
        print(f"Saved annotated image: {annotated_path}")
        
        # Print summary
        total_objects = sum(class_counts.values())
        print(f"\nDetection Summary for {image_path}:")
        print(f"Total objects detected: {total_objects}")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        print("-" * 50)
    
    def process_single_image(self, image_path):
        """
        Process a single image
        
        Args:
            image_path (str): Path to the image file
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist")
            return
        
        # Create output directory
        output_dir = self.create_output_folders()
        
        # Process the image
        self.detect_and_crop_objects(image_path, output_dir)
        
        print(f"\nProcessing complete! Check the '{output_dir}' folder for results.")
    
    def process_multiple_images(self, image_folder):
        """
        Process multiple images from a folder
        
        Args:
            image_folder (str): Path to folder containing images
        """
        if not os.path.exists(image_folder):
            print(f"Error: Folder {image_folder} does not exist")
            return
        
        # Supported image formats
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        # Find all image files
        image_files = []
        for file in os.listdir(image_folder):
            if file.lower().endswith(supported_formats):
                image_files.append(os.path.join(image_folder, file))
        
        if not image_files:
            print(f"No supported image files found in {image_folder}")
            return
        
        print(f"Found {len(image_files)} image(s) to process")
        
        # Create output directory
        output_dir = self.create_output_folders()
        
        # Process each image
        for image_path in image_files:
            self.detect_and_crop_objects(image_path, output_dir)
        
        print(f"\nAll images processed! Check the '{output_dir}' folder for results.")

def main():
    """
    Main function to handle command line arguments and run the detection system
    """
    parser = argparse.ArgumentParser(description='Object Detection and Cropping System')
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to input image file or folder containing images')
    parser.add_argument('--model', '-m', default='yolov8n.pt',
                       help='Path to YOLO model file (default: yolov8n.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold for detection (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize the detector
    detector = ObjectDetectionCropper(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Check if input is a file or folder
    if os.path.isfile(args.input):
        print("Processing single image...")
        detector.process_single_image(args.input)
    elif os.path.isdir(args.input):
        print("Processing multiple images from folder...")
        detector.process_multiple_images(args.input)
    else:
        print(f"Error: {args.input} is not a valid file or folder")

if __name__ == "__main__":
    main()

# Example usage functions (uncomment to use directly in script):

def example_single_image():
    """Example: Process a single image"""
    detector = ObjectDetectionCropper(confidence_threshold=0.3)
    detector.process_single_image("path/to/your/image.jpg")

def example_multiple_images():
    """Example: Process multiple images from a folder"""
    detector = ObjectDetectionCropper(confidence_threshold=0.3)
    detector.process_multiple_images("path/to/your/image/folder")

# Uncomment one of these to run directly:
# example_single_image()
# example_multiple_images()