import cv2
import mediapipe as mp
import numpy as np
import math

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                        max_num_hands=2,
                                        min_detection_confidence=0.7,
                                        min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.gestures = {
            "NONE": "No gesture",
            "FIST": "Fist",
            "OPEN_PALM": "Open Palm",
            "POINTING": "Pointing",
            "PEACE": "Peace Sign",
            "THUMBS_UP": "Thumbs Up",
            "DRAWING": "Drawing Mode",
            "OBJECT_MANIPULATION": "Object Manipulation",
            "SOUND_EFFECTS": "Sound Effects",
            "ANIMATIONS": "Animations"
        }
        self.current_gesture = "NONE"

    def detect_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return image, results

    def find_position(self, image, hand_landmarks, hand_no=0):
        landmark_list = []
        image_height, image_width, _ = image.shape
        if hand_landmarks:
            my_hand = hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                cx, cy = int(lm.x * image_width), int(lm.y * image_height)
                landmark_list.append([id, cx, cy])
        return landmark_list

    def calculate_finger_angles(self, landmark_list):
        angles = []
        finger_joints = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20]
        ]
        if len(landmark_list) > 20:
            for finger in finger_joints:
                if all(idx < len(landmark_list) for idx in [finger[0], finger[1], finger[2]]):
                    a = landmark_list[finger[0]][1:]
                    b = landmark_list[finger[1]][1:]
                    c = landmark_list[finger[2]][1:]
                    angle = self.calculate_angle(a, b, c)
                    angles.append(angle)
        return angles

    def calculate_angle(self, a, b, c):
        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        if magnitude_ba * magnitude_bc == 0:
            return 0
        angle = math.acos(dot_product / (magnitude_ba * magnitude_bc))
        angle = math.degrees(angle)
        return angle

    def detect_gesture(self, landmark_list):
        if not landmark_list or len(landmark_list) < 21:
            return "NONE"
        fingertips = [landmark_list[4], landmark_list[8], landmark_list[12], landmark_list[16], landmark_list[20]]
        finger_bases = [landmark_list[1], landmark_list[5], landmark_list[9], landmark_list[13], landmark_list[17]]
        fingers_extended = []
        thumb_extended = fingertips[0][1] < finger_bases[0][1]
        fingers_extended.append(thumb_extended)
        for i in range(1, 5):
            finger_extended = fingertips[i][2] < landmark_list[finger_bases[i][0]][2]
            fingers_extended.append(finger_extended)
        if not any(fingers_extended):
            return "FIST"
        elif all(fingers_extended):
            return "OPEN_PALM"
        elif fingers_extended[1] and not any(fingers_extended[2:]):
            return "POINTING"
        elif fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]):
            return "PEACE"
        elif not fingers_extended[1:4] and fingers_extended[0] and fingers_extended[4]:
            return "THUMBS_UP"
        elif fingers_extended[0] and fingers_extended[1] and not any(fingers_extended[2:]):
            return "PINCH"
        elif fingers_extended[0] and fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]):
            return "OK_SIGN"
        elif fingers_extended[0] and not any(fingers_extended[1:4]) and fingers_extended[4]:
            return "THUMBS_DOWN"
        elif fingers_extended[1] and fingers_extended[2] and fingers_extended[3] and not fingers_extended[4]:
            return "ROCK_SIGN"
        elif fingers_extended[0] and fingers_extended[1] and fingers_extended[2] and fingers_extended[3] and fingers_extended[4]:
            return "SPIDERMAN_SIGN"
        elif fingers_extended[0] and fingers_extended[1] and not any(fingers_extended[2:4]) and fingers_extended[4]:
            return "PHONE_GESTURE"
        return "NONE"

    def process_frame(self, frame):
        frame, results = self.detect_hands(frame)
        if results.multi_hand_landmarks:
            landmark_list = self.find_position(frame, results.multi_hand_landmarks)
            self.current_gesture = self.detect_gesture(landmark_list)
            cv2.putText(frame,
                        self.gestures[self.current_gesture],
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)
            if self.current_gesture == "DRAWING":
                index_tip = landmark_list[8][1:]
                cv2.circle(frame, tuple(index_tip), 5, (0, 0, 255), -1)
                if hasattr(self, 'prev_point'):
                    cv2.line(frame, self.prev_point, tuple(index_tip), (0, 0, 255), 2)
                self.prev_point = tuple(index_tip)
            else:
                if hasattr(self, 'prev_point'):
                    self.prev_point = None
            if self.current_gesture == "OBJECT_MANIPULATION":
                if not hasattr(self, 'virtual_object'):
                    self.virtual_object = {'position': (200, 200), 'size': 50}
                index_tip = landmark_list[8][1:]
                if math.dist(index_tip, self.virtual_object['position']) < 50:
                    self.virtual_object['position'] = index_tip
                thumb_tip = landmark_list[4][1:]
                distance = math.dist(thumb_tip, index_tip)
                self.virtual_object['size'] = max(20, min(100, distance))
                cv2.circle(frame, self.virtual_object['position'], self.virtual_object['size'], (255, 0, 0), 2)
        return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize hand gesture detector
    detector = HandGestureDetector()
    
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
        
        # Display the frame
        cv2.imshow("Hand Gesture Detection", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()