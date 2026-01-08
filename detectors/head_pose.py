"""
Head Pose Estimation Module
============================
Estimates 3D head orientation (Pitch, Yaw, Roll) using MediaPipe Face Landmarker
and Perspective-n-Point (PnP) algorithm.

Mathematical Background:
-----------------------
The PnP problem solves for the rotation and translation vectors that map
3D object points to 2D image points. We use cv2.solvePnP with a generic
3D face model and detected facial landmarks to estimate head orientation.

Euler Angles:
- Pitch: Nodding motion (up/down rotation around X-axis)
- Yaw: Shaking head (left/right rotation around Y-axis)
- Roll: Tilting head (rotation around Z-axis)
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import config
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HeadPoseEstimator:
    """
    Estimates head pose using MediaPipe Face Landmarker and cv2.solvePnP.
    
    This class detects facial landmarks and maps them to a 3D face model
    to compute the head's orientation in 3D space.
    """
    
    def __init__(self) -> None:
        """
        Initialize MediaPipe Face Landmarker and camera matrix for PnP.
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
        
        # 3D model points (generic face model in world coordinates)
        self.model_points = np.array(config.FACE_3D_MODEL, dtype=np.float64)
        
        # Camera internals (assuming centered principal point)
        self.focal_length = config.FRAME_WIDTH
        self.camera_center = (config.FRAME_WIDTH / 2, config.FRAME_HEIGHT / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Assuming no lens distortion
        self.dist_coeffs = np.zeros((4, 1))
    
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
        
    def _extract_landmarks(self, face_landmarks, img_width: int, img_height: int) -> np.ndarray:
        """
        Extract 2D image points from MediaPipe landmarks.
        
        Args:
            face_landmarks: MediaPipe face landmark list
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            2D numpy array of landmark positions
        """
        image_points = np.array([
            (face_landmarks[config.NOSE_TIP].x * img_width,
             face_landmarks[config.NOSE_TIP].y * img_height),
            (face_landmarks[config.CHIN].x * img_width,
             face_landmarks[config.CHIN].y * img_height),
            (face_landmarks[config.LEFT_EYE_LEFT].x * img_width,
             face_landmarks[config.LEFT_EYE_LEFT].y * img_height),
            (face_landmarks[config.RIGHT_EYE_RIGHT].x * img_width,
             face_landmarks[config.RIGHT_EYE_RIGHT].y * img_height),
            (face_landmarks[config.LEFT_MOUTH].x * img_width,
             face_landmarks[config.LEFT_MOUTH].y * img_height),
            (face_landmarks[config.RIGHT_MOUTH].x * img_width,
             face_landmarks[config.RIGHT_MOUTH].y * img_height)
        ], dtype=np.float64)
        
        return image_points
    
    def _rotation_matrix_to_euler_angles(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (Pitch, Yaw, Roll).
        
        Uses the Rodrigues formula to convert rotation vector to rotation matrix,
        then extracts Euler angles using the ZYX convention.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        # Convert from radians to degrees
        return (
            np.degrees(pitch),
            np.degrees(yaw),
            np.degrees(roll)
        )
    
    def get_pose(self, image: np.ndarray) -> Tuple[Optional[Tuple[float, float, float]], np.ndarray]:
        """
        Estimate head pose from input image.
        
        Process:
        1. Convert image to RGB for MediaPipe
        2. Detect facial landmarks
        3. Extract 2D landmark positions
        4. Solve PnP to get rotation and translation vectors
        5. Convert rotation to Euler angles
        6. Draw pose visualization on image
        
        Args:
            image: Input BGR image from camera
            
        Returns:
            Tuple of ((pitch, yaw, roll), annotated_image)
            Returns (None, original_image) if no face detected
        """
        img_h, img_w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face landmarks
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None, image
        
        face_landmarks = detection_result.face_landmarks[0]
        
        # Extract 2D image points
        image_points = self._extract_landmarks(face_landmarks, img_w, img_h)
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, image
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles
        pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        # Optional: Draw nose direction line (disabled by default for cleaner UI)
        # Uncomment below to enable nose direction visualization
        # nose_tip_2d = image_points[0].astype(int)
        # nose_end_point3D = np.array([[0.0, 0.0, 1000.0]])
        # nose_end_point2D, _ = cv2.projectPoints(
        #     nose_end_point3D, rotation_vector, translation_vector,
        #     self.camera_matrix, self.dist_coeffs
        # )
        # p1, p2 = tuple(nose_tip_2d), tuple(nose_end_point2D[0][0].astype(int))
        # cv2.line(image, p1, p2, (255, 0, 0), 3)
        # cv2.circle(image, p1, 5, (0, 255, 255), -1)
        
        return (pitch, yaw, roll), image
    
    def __del__(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, 'detector'):
            self.detector.close()