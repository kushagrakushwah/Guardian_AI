"""
Drowsiness Detection Module
============================
Detects driver drowsiness using Eye Aspect Ratio (EAR) metric.

Mathematical Background:
-----------------------
Eye Aspect Ratio (EAR) is computed using 6 eye landmarks:
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
Where:
- p1, p4: Horizontal eye corners (left and right)
- p2, p3, p5, p6: Vertical eye landmarks (top and bottom pairs)

The EAR value is approximately constant when the eye is open,
but rapidly falls to zero when the eye is closed.

Reference: Soukupová and Čech, "Real-Time Eye Blink Detection using 
Facial Landmarks" (2016)
"""

import cv2 
import numpy as np
from collections import deque
from typing import Optional, Tuple
import config
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class DrowsinessDetector:
    """
    Detects drowsiness by monitoring Eye Aspect Ratio (EAR) over time.
    
    Uses a moving average buffer to smooth EAR values and reduce
    false positives from natural blinking.
    """
    
    def __init__(self) -> None:
        """
        Initialize MediaPipe Face Landmarker and EAR tracking buffers.
        """
        # Create Face Landmarker with base options
        base_options = python.BaseOptions(
            model_asset_buffer=self._download_model()
        )
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=config.MEDIAPIPE_MAX_FACES,
            min_face_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Buffers for smoothing EAR values (moving average)
        self.left_ear_buffer: deque = deque(maxlen=config.EAR_BUFFER_SIZE)
        self.right_ear_buffer: deque = deque(maxlen=config.EAR_BUFFER_SIZE)
        
        # Counter for consecutive frames with low EAR
        self.drowsy_frames: int = 0
        
        # Current status
        self.current_status: str = "ALERT"
    
    def _download_model(self) -> bytes:
        """
        Download MediaPipe Face Landmarker model.
        
        Returns:
            Model file as bytes
        """
        import urllib.request
        import os
        
        model_path = "face_landmarker.task"
        
        if not os.path.exists(model_path):
            print("Downloading MediaPipe Face Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("✓ Model downloaded successfully")
        
        with open(model_path, "rb") as f:
            return f.read()
        
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(point1 - point2)
    
    def calculate_EAR(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for given eye landmarks.
        
        The EAR formula quantifies eye openness. It approaches zero
        when the eye is closed and remains constant (around 0.3) when open.
        
        Formula:
            EAR = (vertical_1 + vertical_2) / (2 * horizontal)
            
        Where:
            vertical_1 = distance between landmarks p2 and p6
            vertical_2 = distance between landmarks p3 and p5
            horizontal = distance between landmarks p1 and p4
        
        Args:
            eye_landmarks: Array of 6 eye landmark coordinates (x, y)
                          Ordered as: [p1, p2, p3, p4, p5, p6]
        
        Returns:
            Eye Aspect Ratio (float)
        """
        # Vertical eye distances
        vertical_1 = self._euclidean_distance(eye_landmarks[1], eye_landmarks[5])
        vertical_2 = self._euclidean_distance(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal eye distance
        horizontal = self._euclidean_distance(eye_landmarks[0], eye_landmarks[3])
        
        # EAR calculation
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        
        return ear
    
    def _extract_eye_landmarks(self, face_landmarks, img_width: int, img_height: int, 
                               eye_indices: list) -> np.ndarray:
        """
        Extract eye landmark coordinates from MediaPipe results.
        
        Args:
            face_landmarks: MediaPipe face landmark list
            img_width: Image width in pixels
            img_height: Image height in pixels
            eye_indices: List of landmark indices for the eye
            
        Returns:
            Array of eye landmark coordinates
        """
        eye_points = []
        for idx in eye_indices:
            landmark = face_landmarks[idx]
            x = int(landmark.x * img_width)
            y = int(landmark.y * img_height)
            eye_points.append([x, y])
        
        return np.array(eye_points, dtype=np.float64)
    
    def detect_drowsiness(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Detect drowsiness in the input image.
        
        Process:
        1. Detect face landmarks
        2. Extract left and right eye landmarks
        3. Calculate EAR for both eyes
        4. Add to moving average buffer
        5. Check if average EAR is below threshold
        6. Update drowsy frame counter
        7. Determine status (ALERT/DROWSY)
        
        Args:
            image: Input BGR image from camera
            
        Returns:
            Tuple of (status, average_EAR, annotated_image)
        """
        img_h, img_w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face landmarks
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return "NO_FACE", 0.0, image
        
        face_landmarks = detection_result.face_landmarks[0]
        
        # Extract eye landmarks
        left_eye = self._extract_eye_landmarks(
            face_landmarks, img_w, img_h, config.LEFT_EYE_INDICES
        )
        right_eye = self._extract_eye_landmarks(
            face_landmarks, img_w, img_h, config.RIGHT_EYE_INDICES
        )
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_EAR(left_eye)
        right_ear = self.calculate_EAR(right_eye)
        
        # Add to buffers
        self.left_ear_buffer.append(left_ear)
        self.right_ear_buffer.append(right_ear)
        
        # Calculate average EAR (from both eyes and buffer history)
        avg_ear = (np.mean(self.left_ear_buffer) + np.mean(self.right_ear_buffer)) / 2.0
        
        # Check drowsiness
        self.current_status = self.check_status(avg_ear)
        
        # Draw eye landmarks for visualization (optional - disable for cleaner look)
        # Uncomment to show eye tracking points
        # for point in left_eye:
        #     cv2.circle(image, tuple(point.astype(int)), 2, (0, 255, 0), -1)
        # for point in right_eye:
        #     cv2.circle(image, tuple(point.astype(int)), 2, (0, 255, 0), -1)
        
        return self.current_status, avg_ear, image
    
    def check_status(self, avg_ear: float) -> str:
        """
        Determine alertness status based on average EAR.
        
        Uses consecutive frame counting to avoid false positives
        from natural blinking. Drowsiness is confirmed only after
        EAR remains below threshold for BLINK_CONSEC_FRAMES.
        
        Args:
            avg_ear: Average Eye Aspect Ratio
            
        Returns:
            Status string: "ALERT" or "DROWSY"
        """
        if avg_ear < config.EAR_THRESHOLD:
            self.drowsy_frames += 1
            
            if self.drowsy_frames >= config.BLINK_CONSEC_FRAMES:
                return "DROWSY"
        else:
            self.drowsy_frames = 0
            return "ALERT"
        
        return "ALERT"
    
    def reset(self) -> None:
        """
        Reset all buffers and counters.
        Useful for starting a new monitoring session.
        """
        self.left_ear_buffer.clear()
        self.right_ear_buffer.clear()
        self.drowsy_frames = 0
        self.current_status = "ALERT"
    
    def __del__(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, 'detector'):
            self.detector.close()