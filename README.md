# OpenCV Object Detection and Measurement

A real-time object detection and measurement system using OpenCV that can detect objects through a webcam and measure their dimensions in pixels.

## Features

- **Real-time Object Detection**: Captures video from webcam and detects objects in real-time
- **Contour Analysis**: Uses OpenCV's contour detection to identify object boundaries
- **Dimension Measurement**: Calculates and displays object length and height in pixels
- **Visual Feedback**: Draws bounding boxes around detected objects with measurement labels
- **Noise Filtering**: Filters out small noise objects using area threshold

## Requirements

- Python 3.6+
- OpenCV (cv2)
- Webcam

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Phani69/opencv-object-measurement.git
   cd opencv-object-measurement
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python
   ```

3. **Run the application**:
   ```bash
   python Object_Detection.py
   ```

## Usage

1. **Start the application**: Run the script and it will open your webcam
2. **Position objects**: Place objects in front of the camera
3. **View measurements**: The system will display bounding boxes with length and height measurements
4. **Exit**: Press 'q' to quit the application

## How it Works

### Object Detection Process

1. **Video Capture**: Captures frames from the webcam at 1280x720 resolution
2. **Image Preprocessing**:
   - Converts frames to grayscale
   - Applies Gaussian blur to reduce noise
   - Uses binary thresholding to segment objects from background
3. **Contour Detection**: Finds external contours of objects
4. **Measurement**: Calculates minimum area rectangle for each object
5. **Display**: Draws bounding boxes and displays dimensions

### Key Parameters

- **Resolution**: 1280x720 (HD)
- **Blur Kernel**: 5x5 Gaussian blur
- **Threshold**: 160 (binary threshold for object segmentation)
- **Minimum Area**: 500 pixels (filters out noise)

## Code Structure

```python
# Main components:
- Video capture setup
- Image preprocessing pipeline
- Contour detection and analysis
- Dimension calculation
- Visual output
```

## Output

The application displays:
- Live video feed with detected objects
- Green bounding boxes around objects
- Text showing length and height in pixels
- Resized display (1200x800) for better visibility

## Customization

You can modify the following parameters in `Object_Detection.py`:

- **Camera resolution**: Change `cap.set()` values
- **Threshold value**: Adjust the binary threshold (currently 160)
- **Minimum area**: Change the noise filter threshold (currently 500)
- **Display size**: Modify the resize dimensions

## Troubleshooting

### Common Issues

1. **Camera not found**: Ensure your webcam is connected and not in use by another application
2. **Poor detection**: Adjust lighting conditions or threshold values
3. **High CPU usage**: Reduce resolution or increase blur kernel size

### Performance Tips

- Use good lighting for better object detection
- Keep objects at a reasonable distance from the camera
- Close other applications using the webcam

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Submitting pull requests

## License

This project is open source and available under the MIT License.

## Author

Created by Phani69

## Repository

GitHub: https://github.com/Phani69/opencv-object-measurement.git
