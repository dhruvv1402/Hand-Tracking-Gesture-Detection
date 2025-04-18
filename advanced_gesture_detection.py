import cv2
import mediapipe as mp
import numpy as np
from hand_gesture_detection import HandGestureDetector
from gesture_utils import detect_advanced_gestures, get_finger_directions

class AdvancedGestureDetector(HandGestureDetector):
    def __init__(self):
        super().__init__()
        # Additional gestures
        self.advanced_gestures = {
            "PINCH": "Pinch Gesture",
            "OK_SIGN": "OK Sign",
            "ROCK_SIGN": "Rock Sign",
            "THUMBS_DOWN": "Thumbs Down",
            "SPIDERMAN": "Spider-Man Sign",
            "PHONE": "Phone Gesture",
            "UNKNOWN": "Unknown Gesture"
        }
        self.gestures.update(self.advanced_gestures)
        
        # Track gesture history for smoother detection
        self.gesture_history = []
        self.history_length = 5
        
        # Track finger directions
        self.finger_directions = {}
    
    def process_frame(self, frame):
        # Get original processing from parent class
        frame, results = self.detect_hands(frame)
        
        # If hands are detected
        if results.multi_hand_landmarks:
            # Get landmarks for the first hand
            landmark_list = self.find_position(frame, results.multi_hand_landmarks)
            
            # Basic gesture detection
            basic_gesture = self.detect_gesture(landmark_list)
            
            # Advanced gesture detection
            advanced_gesture = detect_advanced_gestures(landmark_list)
            
            # Determine final gesture (prioritize advanced gestures)
            if advanced_gesture != "UNKNOWN":
                self.current_gesture = advanced_gesture
            else:
                self.current_gesture = basic_gesture
            
            # Update gesture history for smoothing
            self.gesture_history.append(self.current_gesture)
            if len(self.gesture_history) > self.history_length:
                self.gesture_history.pop(0)
            
            # Get the most common gesture in history
            if self.gesture_history:
                from collections import Counter
                most_common = Counter(self.gesture_history).most_common(1)[0][0]
                self.current_gesture = most_common
            
            # Get finger directions
            image_height, image_width, _ = frame.shape
            self.finger_directions = get_finger_directions(landmark_list, image_width, image_height)
            
            # Display gesture name
            cv2.putText(frame, 
                        self.gestures[self.current_gesture], 
                        (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA)
            
            # Display finger directions
            y_pos = 90
            for finger, direction in self.finger_directions.items():
                text = f"{finger}: {direction}"
                cv2.putText(frame, 
                            text, 
                            (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 0, 0), 
                            1, 
                            cv2.LINE_AA)
                y_pos += 30
        
        return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize advanced hand gesture detector
    detector = AdvancedGestureDetector()
    
    # Set up window
    cv2.namedWindow("Advanced Hand Gesture Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Advanced Hand Gesture Detection", 1280, 720)
    
    while True:
        # Read frame from webcam
        success, frame = cap.read()
        
        if not success:
            print("Failed to capture video")
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Process the frame for hand detection and gesture recognition
        frame = detector.process_frame(frame)
        
        # Add instructions
        cv2.putText(frame, 
                    "Press 'q' to quit", 
                    (frame.shape[1] - 200, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow("Advanced Hand Gesture Detection", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()