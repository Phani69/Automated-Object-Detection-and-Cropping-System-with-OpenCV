#  Automated Object Detection and Cropping System with OpenCV

A comprehensive object detection and cropping system using YOLOv8 and OpenCV that can detect objects in images and automatically crop them into separate files organized by class.

## Features

- **YOLOv8 Object Detection**: Uses state-of-the-art YOLOv8 model for accurate object detection
- **Multi-Class Support**: Detects 80 different object classes from the COCO dataset
- **Automatic Cropping**: Extracts detected objects as individual image files
- **Organized Output**: Saves cropped objects in class-specific folders
- **Batch Processing**: Process single images or entire folders
- **GUI Interface**: User-friendly graphical interface for easy operation
- **Command Line Interface**: Script-based processing for automation
- **Confidence Control**: Adjustable confidence threshold for detection accuracy
- **Annotated Output**: Generates annotated images with bounding boxes

## Supported Object Classes

The system can detect 80 different object classes including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, boat
- **Animals**: dog, cat, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objects**: chair, couch, bed, table, tv, laptop, phone, book, clock
- **Food**: apple, banana, orange, pizza, cake, hot dog, sandwich
- **And many more...**

## Requirements

- Python 3.8+
- OpenCV (cv2)
- Ultralytics (YOLOv8)
- NumPy
- Matplotlib
- Pillow
- scikit-image
- tkinter (for GUI)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Phani69/opencv-object-measurement.git
cd opencv-object-measurement
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Or use the provided activation script:
./activate_env.sh
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model
The project includes a pre-trained YOLOv8n model (`yolov8n.pt`). If you need a different model:
```bash
# Download YOLOv8n (nano) - already included
# Download YOLOv8s (small)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# Download YOLOv8m (medium)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# Download YOLOv8l (large)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
```

## Usage

### GUI Interface (Recommended)

1. **Start the GUI**:
   ```bash
   python gui.py
   ```

2. **Configure Settings**:
   - Adjust confidence threshold using the slider
   - Select single image or folder of images

3. **Process Images**:
   - Click "Process Images" to start detection
   - Monitor progress in the status window

### Command Line Interface

#### Process Single Image
```bash
python Object_Detection.py --input path/to/image.jpg
```

#### Process Multiple Images
```bash
python Object_Detection.py --input path/to/image/folder
```

#### Custom Settings
```bash
# With custom confidence threshold
python Object_Detection.py --input image.jpg --confidence 0.3

# With custom model
python Object_Detection.py --input image.jpg --model yolov8s.pt
```

#### Command Line Options
- `--input, -i`: Path to input image file or folder (required)
- `--model, -m`: Path to YOLO model file (default: yolov8n.pt)
- `--confidence, -c`: Confidence threshold (default: 0.5)

## Output Structure

The system creates timestamped output folders with the following structure:

```
detected_objects_20241201_143022/
├── person/
│   ├── image1_person_001_conf0.85.jpg
│   └── image1_person_002_conf0.92.jpg
├── car/
│   ├── image1_car_001_conf0.78.jpg
│   └── image2_car_001_conf0.89.jpg
├── dog/
│   └── image1_dog_001_conf0.95.jpg
└── annotated_image1.jpg
```

### Output Files
- **Class folders**: Each detected object class gets its own folder
- **Cropped images**: Individual object images with confidence scores
- **Annotated images**: Original images with bounding boxes and labels

## How It Works

### Detection Process

1. **Model Loading**: Loads pre-trained YOLOv8 model
2. **Image Processing**: Reads and preprocesses input images
3. **Object Detection**: Runs YOLO inference to detect objects
4. **Confidence Filtering**: Filters detections based on confidence threshold
5. **Object Cropping**: Extracts bounding box regions as separate images
6. **File Organization**: Saves cropped objects in class-specific folders
7. **Annotation**: Creates annotated versions with bounding boxes

### Key Components

- **YOLOv8 Model**: State-of-the-art object detection model
- **OpenCV**: Image processing and file I/O
- **Ultralytics**: YOLO model interface and inference
- **Tkinter**: GUI framework for user interface

## Configuration

### Confidence Threshold
- **Default**: 0.5 (50% confidence)
- **Range**: 0.1 to 1.0
- **Lower values**: More detections, potentially more false positives
- **Higher values**: Fewer detections, higher accuracy

### Model Selection
- **YOLOv8n**: Fastest, smallest model (default)
- **YOLOv8s**: Small model, good balance of speed/accuracy
- **YOLOv8m**: Medium model, higher accuracy
- **YOLOv8l**: Large model, highest accuracy

## Performance

### Speed vs Accuracy Trade-off
- **YOLOv8n**: ~6ms inference time, good for real-time applications
- **YOLOv8s**: ~12ms inference time, better accuracy
- **YOLOv8m**: ~25ms inference time, high accuracy
- **YOLOv8l**: ~45ms inference time, highest accuracy


## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `yolov8n.pt` is in the project directory
2. **Import errors**: Activate virtual environment and install requirements
3. **CUDA errors**: Install PyTorch with CUDA support for GPU acceleration
4. **Memory issues**: Use smaller model or reduce batch size



```

## Contributing

Contributions are welcome! Please feel free to:

- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is open source and available under the MIT License.

## Author

Created by Phani69

## Repository

GitHub: https://github.com/Phani69/opencv-object-measurement.git

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [COCO Dataset](https://cocodataset.org/) for object classes
