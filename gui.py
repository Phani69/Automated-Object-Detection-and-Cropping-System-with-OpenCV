import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
from ultralytics import YOLO
import numpy as np
from datetime import datetime
from pathlib import Path
import threading

class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection and Cropping System")
        self.root.geometry("600x500")
        
        self.model = None
        self.confidence_threshold = 0.5
        
        # COCO class names
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
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Object Detection and Cropping System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Model selection frame
        model_frame = ttk.LabelFrame(main_frame, text="Model Settings", padding="10")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(model_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(model_frame, from_=0.1, to=1.0, variable=self.confidence_var,
                                   orient=tk.HORIZONTAL, length=200)
        confidence_scale.grid(row=0, column=1, padx=(10, 0))
        
        self.confidence_label = ttk.Label(model_frame, text="0.5")
        self.confidence_label.grid(row=0, column=2, padx=(10, 0))
        
        # Update confidence label
        confidence_scale.configure(command=self.update_confidence_label)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Input Selection", padding="10")
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Single image button
        self.single_btn = ttk.Button(file_frame, text="Select Single Image", 
                                   command=self.select_single_image, width=20)
        self.single_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Multiple images button
        self.folder_btn = ttk.Button(file_frame, text="Select Image Folder", 
                                   command=self.select_image_folder, width=20)
        self.folder_btn.grid(row=0, column=1)
        
        # Selected path display
        self.path_var = tk.StringVar(value="No file/folder selected")
        path_label = ttk.Label(file_frame, textvariable=self.path_var, wraplength=400)
        path_label.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="Process Images", 
                                    command=self.process_images, state="disabled",
                                    style="Accent.TButton")
        self.process_btn.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status text
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.status_text = tk.Text(status_frame, height=15, width=70)
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
    
    def load_model(self):
        """Load YOLO model"""
        try:
            self.log_status("Loading YOLO model...")
            self.model = YOLO('yolov8n.pt')  # This will download the model if not present
            self.log_status("Model loaded successfully!")
        except Exception as e:
            self.log_status(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")
    
    def update_confidence_label(self, value):
        """Update confidence threshold label"""
        self.confidence_label.config(text=f"{float(value):.2f}")
        self.confidence_threshold = float(value)
    
    def log_status(self, message):
        """Add message to status text"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def select_single_image(self):
        """Select single image file"""
        filetypes = (
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Select an image',
            filetypes=filetypes
        )
        
        if filename:
            self.selected_path = filename
            self.path_var.set(f"Selected: {filename}")
            self.process_btn.config(state="normal")
            self.log_status(f"Selected image: {filename}")
    
    def select_image_folder(self):
        """Select folder containing images"""
        folder = filedialog.askdirectory(title='Select folder with images')
        
        if folder:
            self.selected_path = folder
            self.path_var.set(f"Selected: {folder}")
            self.process_btn.config(state="normal")
            self.log_status(f"Selected folder: {folder}")
    
    def create_output_folders(self, base_path="detected_objects"):
        """Create output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{base_path}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.log_status(f"Created output directory: {output_dir}")
        return output_dir
    
    def detect_and_crop_objects(self, image_path, output_base_dir):
        """Detect and crop objects from image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self.log_status(f"Error: Could not load image {image_path}")
            return
        
        self.log_status(f"Processing: {os.path.basename(image_path)}")
        
        # Run inference
        results = self.model(image)
        detections = results[0].boxes
        
        if detections is None or len(detections) == 0:
            self.log_status("No objects detected")
            return
        
        class_counts = {}
        
        # Process each detection
        for detection in detections:
            confidence = float(detection.conf)
            class_id = int(detection.cls)
            
            if confidence < self.confidence_threshold:
                continue
            
            class_name = self.class_names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Get bounding box
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            
            # Create class folder
            class_folder = os.path.join(output_base_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            
            # Crop and save
            cropped_object = image[y1:y2, x1:x2]
            input_filename = Path(image_path).stem
            crop_filename = f"{input_filename}_{class_name}_{class_counts[class_name]:03d}_conf{confidence:.2f}.jpg"
            crop_path = os.path.join(class_folder, crop_filename)
            
            cv2.imwrite(crop_path, cropped_object)
            self.log_status(f"Saved: {class_name} (conf: {confidence:.2f})")
        
        # Save annotated image
        annotated_image = results[0].plot()
        annotated_path = os.path.join(output_base_dir, f"annotated_{Path(image_path).name}")
        cv2.imwrite(annotated_path, annotated_image)
        
        # Log summary
        total_objects = sum(class_counts.values())
        self.log_status(f"Total objects detected: {total_objects}")
        for class_name, count in class_counts.items():
            self.log_status(f"  {class_name}: {count}")
    
    def process_images_thread(self):
        """Process images in separate thread"""
        try:
            output_dir = self.create_output_folders()
            
            if os.path.isfile(self.selected_path):
                # Single image
                self.detect_and_crop_objects(self.selected_path, output_dir)
            else:
                # Multiple images
                supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
                image_files = [
                    os.path.join(self.selected_path, f) 
                    for f in os.listdir(self.selected_path)
                    if f.lower().endswith(supported_formats)
                ]
                
                if not image_files:
                    self.log_status("No supported image files found")
                    return
                
                self.log_status(f"Found {len(image_files)} image(s) to process")
                
                for image_path in image_files:
                    self.detect_and_crop_objects(image_path, output_dir)
            
            self.log_status(f"\nProcessing complete! Results saved in: {output_dir}")
            messagebox.showinfo("Complete", f"Processing complete!\nResults saved in: {output_dir}")
            
        except Exception as e:
            self.log_status(f"Error during processing: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:
            # Re-enable UI
            self.root.after(0, self.processing_finished)
    
    def processing_finished(self):
        """Called when processing is finished"""
        self.progress.stop()
        self.process_btn.config(state="normal")
        self.single_btn.config(state="normal")
        self.folder_btn.config(state="normal")
    
    def process_images(self):
        """Start image processing"""
        if not hasattr(self, 'selected_path'):
            messagebox.showwarning("Warning", "Please select an image or folder first")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        # Disable UI during processing
        self.process_btn.config(state="disabled")
        self.single_btn.config(state="disabled")
        self.folder_btn.config(state="disabled")
        
        self.progress.start()
        self.log_status("Starting processing...")
        
        # Start processing in separate thread
        thread = threading.Thread(target=self.process_images_thread)
        thread.daemon = True
        thread.start()

def main():
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()