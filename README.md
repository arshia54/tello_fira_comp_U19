AprilTag Detection System

A powerful AprilTag detection system with automatic detector selection, adaptive image processing, and intelligent tracking.

Features

- Three detection modes: FAST, PRECISE, LOW_LIGHT
- Automatic detector selection based on lighting conditions
- Five-factor confidence calculation for each detection
- Kalman filter for smooth tracking and position prediction
- Real-time performance metrics display
- Webcam support (no Tello drone required)

Installation and Run

Step 1 - Install dependencies:
pip install opencv-python pupil-apriltags numpy scipy

Step 2 - Run the program:
python myat.py

Controls

Key Q - Exit program
Key R - Reset tracking system
Key S - Show statistics in console

Detection Modes

FAST mode - Speed priority, lower resolution for faster processing
PRECISE mode - Accuracy priority, full resolution for higher precision
LOW_LIGHT mode - Poor lighting, enhanced contrast and noise reduction

How Detection Works

1. Camera captures image
2. System assesses lighting conditions (low light, normal, high contrast)
3. Image preprocessing (CLAHE, denoise, sharpen) based on lighting
4. Best detector selected automatically
5. AprilTag detection runs
6. Confidence calculated from 5 factors
7. Kalman filter smooths position and predicts movement
8. Validated results returned

Confidence Factors

Tag structure (20%) - Valid corners present
Area (20%) - Appropriate tag size in frame
Edge quality (25%) - Sharpness of tag edges
Pose stability (15%) - Reasonable distance and angle
Historical consistency (20%) - Matches previous detections

Display Information

Colored circle on tag - Green (high confidence), Yellow (medium), Red (low)
ID - Tag identification number
Pos - 3D position in mm (X, Y, Z)
Ang - Angle in degrees
Conf - Confidence value from 0 to 1
Det - Active detector name

Performance Metrics (top-left corner)

Frames - Total processed frames
Success - Detection success rate as percentage
Avg Time - Average processing time in milliseconds
Detector - Currently active detector
Brightness - Average image brightness

Output Format

get_tags() returns a list where each element contains:
[
    tag_id,      # Tag identification number
    x, y, z,     # 3D position in millimeters
    angle,       # Angle in degrees
    center_x,    # X center in pixels
    center_y,    # Y center in pixels
    confidence,  # Confidence value from 0 to 1
    detector     # Detector name used (FAST, PRECISE, or LOW_LIGHT)
]

Troubleshooting

Tag not detected - Increase lighting, move closer to tag, check print quality
Slow detection - System should auto-select FAST mode, or lower camera resolution
Low confidence - Improve lighting, clean the tag, let system switch to PRECISE mode
Jittery tracking - Press R key to reset tracking system

Configuration

Edit these values at the top of myat.py:

frame_size = [960, 720]      - Frame width and height in pixels
april_cm = 1000.0            - Scale factor (mm to cm conversion)
april_focal = [680, 680]     - Camera focal length in pixels
TAG_SIZE = 0.016             - Physical tag size in meters

Detection thresholds you can adjust:

min_confidence = 0.3         - Minimum confidence to accept detection
max_position_mm = 1000       - Maximum reasonable position in mm

Lighting thresholds:

brightness_low = 60          - Below this = low light condition
brightness_high = 200        - Above this = high contrast condition

Sample Output

When you press S key:
{
  "total_frames": 150,
  "success_rate": 0.92,
  "avg_processing_time": 28.5,
  "current_detector": "FAST",
  "detector_performance": {
    "FAST": [120, 0.88],
    "PRECISE": [25, 0.95],
    "LOW_LIGHT": [5, 0.71]
  }
}

Requirements File (requirements.txt)

opencv-python>=4.8.0
pupil-apriltags>=1.1.0
numpy>=1.24.0
scipy>=1.11.0

Notes

- System automatically selects best detector based on lighting and performance
- For best results, keep tag centered in frame
- Uniform lighting gives best detection results
- In low light, system automatically switches to LOW_LIGHT detector

License

MIT License

Author

DayLight_

Version

1.0.0
