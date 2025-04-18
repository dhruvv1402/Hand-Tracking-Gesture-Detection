# Hand Tracking and Gesture Detection

This project implements real-time hand tracking and gesture detection using OpenCV and MediaPipe. It can recognize various hand gestures including:

- Open Palm
- Fist
- Pointing (Index finger)
- Peace Sign
- Thumbs Up
- Thumbs Down
- OK Sign
- Rock Sign
- Pinch Gesture
- Spider-Man Sign
- Phone Gesture

## Features

- Real-time hand detection and tracking
- Multiple gesture recognition
- Visual feedback with landmark visualization
- Simple and intuitive interface

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

Run the main script:

```
python hand_gesture_detection.py
```

- Press 'q' to quit the application
- Position your hand in front of the camera to detect gestures

## Additional Information

This project is designed to be a simple demonstration of hand gesture recognition using computer vision techniques. It can be extended to include more complex gestures or integrated into larger applications requiring gesture-based interactions.

## How It Works

The application uses MediaPipe's hand tracking solution to detect hand landmarks in real-time. These landmarks are then analyzed to determine the position and state of each finger, which is used to recognize specific gestures.

## Customization

You can add more gestures by modifying the `detect_gesture` method in the `HandGestureDetector` class. The current implementation detects basic gestures, but you can extend it to recognize more complex hand movements.

## License

This project is open-source and available for personal and educational use.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)