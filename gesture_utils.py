import numpy as np
import math

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_angle_three_points(a, b, c):
    """Calculate angle between three points with b as the vertex"""
    # Vectors
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    
    # Normalize vectors
    ba_norm = ba / np.linalg.norm(ba)
    bc_norm = bc / np.linalg.norm(bc)
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba_norm, bc_norm)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def is_finger_bent(mcp, pip, dip, tip, threshold=90):
    """Determine if a finger is bent based on joint angles"""
    # Calculate angle at PIP joint
    angle = calculate_angle_three_points(mcp, pip, dip)
    
    # If angle is less than threshold, finger is considered bent
    return angle < threshold

def detect_advanced_gestures(landmarks):
    """Detect more complex gestures based on hand landmarks"""
    if not landmarks or len(landmarks) < 21:
        return "UNKNOWN"
    
    # Extract key points (using landmark indices)
    wrist = landmarks[0][1:]
    thumb_mcp = landmarks[1][1:]
    thumb_tip = landmarks[4][1:]
    index_tip = landmarks[8][1:]
    middle_tip = landmarks[12][1:]
    ring_tip = landmarks[16][1:]
    pinky_tip = landmarks[20][1:]
    
    # Get positions of finger bases (landmarks 5, 9, 13, 17)
    index_mcp = landmarks[5][1:]
    middle_mcp = landmarks[9][1:]
    ring_mcp = landmarks[13][1:]
    pinky_mcp = landmarks[17][1:]
    
    # Calculate distances between fingertips
    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    index_middle_distance = calculate_distance(index_tip, middle_tip)
    thumb_pinky_distance = calculate_distance(thumb_tip, pinky_tip)
    
    # Detect pinch gesture (thumb and index finger close together)
    if thumb_index_distance < 40:  # Threshold can be adjusted
        return "PINCH"
    
    # Detect OK sign (thumb and index form a circle, other fingers extended)
    if thumb_index_distance < 50 and index_middle_distance > 50:
        return "OK_SIGN"
    
    # Detect Rock sign (index and pinky extended, middle and ring curled)
    index_extended = index_tip[1] < index_mcp[1]  # Y-coordinate comparison
    middle_extended = middle_tip[1] < middle_mcp[1]
    ring_extended = ring_tip[1] < ring_mcp[1]
    pinky_extended = pinky_tip[1] < pinky_mcp[1]
    
    if index_extended and pinky_extended and not middle_extended and not ring_extended:
        return "ROCK_SIGN"
    
    # Detect Thumbs Down (only thumb is down, others may be in various positions)
    thumb_down = thumb_tip[1] > thumb_mcp[1]  # Thumb tip is below thumb MCP
    if thumb_down and calculate_distance(thumb_tip, wrist) > 100:  # Ensure thumb is extended downward
        return "THUMBS_DOWN"
    
    # Detect Spider-Man gesture (thumb, pinky, and index extended)
    if index_extended and pinky_extended and not middle_extended and not ring_extended and thumb_pinky_distance > 100:
        return "SPIDERMAN"
    
    # Detect Phone gesture (thumb and pinky extended like a phone)
    if not index_extended and not middle_extended and not ring_extended and pinky_extended:
        thumb_to_ear = calculate_distance(thumb_tip, wrist) > 80  # Thumb extended
        if thumb_to_ear:
            return "PHONE"
    
    return "UNKNOWN"

def get_finger_directions(landmarks, image_width, image_height):
    """Determine the direction each finger is pointing"""
    if not landmarks or len(landmarks) < 21:
        return {}
    
    # Define fingertips and their corresponding MCP joints
    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    mcp_joints = [1, 5, 9, 13, 17]   # Corresponding MCP joints
    
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    directions = {}
    
    for i, (tip_idx, mcp_idx) in enumerate(zip(fingertips, mcp_joints)):
        # Get coordinates
        tip_x, tip_y = landmarks[tip_idx][1], landmarks[tip_idx][2]
        mcp_x, mcp_y = landmarks[mcp_idx][1], landmarks[mcp_idx][2]
        
        # Calculate direction vector
        dir_x = tip_x - mcp_x
        dir_y = tip_y - mcp_y
        
        # Determine primary direction
        if abs(dir_x) > abs(dir_y):
            # Horizontal direction
            direction = "Right" if dir_x > 0 else "Left"
        else:
            # Vertical direction
            direction = "Up" if dir_y < 0 else "Down"  # Y-axis is inverted in image coordinates
        
        directions[finger_names[i]] = direction
    
    return directions