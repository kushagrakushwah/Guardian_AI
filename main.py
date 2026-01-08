"""
GuardianAI - Driver Monitoring System
======================================
Main application module that integrates all components.

Features:
- Real-time drowsiness detection using Eye Aspect Ratio (EAR)
- Head pose estimation for distraction detection
- Dynamic safety scoring system
- Visual dashboard with alerts
- Incident logging and evidence collection

Usage:
    python main.py
    
Controls:
    'q' - Quit application
    'r' - Reset safety score
    's' - Save current snapshot
    'c' - Clear logs
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from detectors.head_pose import HeadPoseEstimator
from detectors.drowsiness import DrowsinessDetector
from utils.logger import IncidentLogger


class GuardianSystem:
    """
    Main driver monitoring system integrating all detection modules.
    
    This class orchestrates the entire monitoring pipeline:
    - Captures video frames
    - Runs drowsiness and distraction detection
    - Maintains safety scoring system
    - Logs incidents and saves evidence
    - Renders real-time dashboard UI
    """
    
    def __init__(self) -> None:
        """
        Initialize all system components and verify camera access.
        """
        print("=" * 60)
        print("GuardianAI - Driver Monitoring System")
        print("=" * 60)
        
        # Initialize components
        print("\n[1/4] Initializing camera...")
        self.camera = self._initialize_camera()
        
        print("[2/4] Loading head pose estimator...")
        self.head_pose_estimator = HeadPoseEstimator()
        
        print("[3/4] Loading drowsiness detector...")
        self.drowsiness_detector = DrowsinessDetector()
        
        print("[4/4] Initializing incident logger...")
        self.logger = IncidentLogger()
        
        # Safety score system
        self.safety_score: float = config.INITIAL_SAFETY_SCORE
        
        # Frame counter for periodic logging
        self.frame_count: int = 0
        self.log_interval: int = 30  # Log every 30 frames
        
        # Last logged event to avoid duplicate logs
        self.last_event: str = "NORMAL"
        
        # Alert state for visual feedback
        self.alert_active: bool = False
        self.alert_message: str = ""
        
        print("\n✓ System initialized successfully!")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'r' - Reset score")
        print("  's' - Save snapshot")
        print("  'c' - Clear logs")
        print("\nStarting monitoring...")
        print("=" * 60)
    
    def _initialize_camera(self) -> cv2.VideoCapture:
        """
        Initialize camera with error handling.
        
        Returns:
            VideoCapture object
            
        Raises:
            RuntimeError: If camera cannot be initialized
        """
        camera = cv2.VideoCapture(config.CAMERA_ID)
        
        if not camera.isOpened():
            raise RuntimeError(
                f"✗ Failed to open camera {config.CAMERA_ID}. "
                "Please check camera connection."
            )
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, config.FPS)
        
        return camera
    
    def _update_safety_score(
        self,
        drowsy: bool,
        distracted: bool
    ) -> None:
        """
        Update safety score based on driver behavior.
        
        Score decreases with violations and slowly regenerates
        during normal behavior.
        
        Args:
            drowsy: Whether driver is drowsy
            distracted: Whether driver is distracted
        """
        if drowsy:
            self.safety_score -= config.DROWSINESS_PENALTY
            self.alert_message = "⚠ DROWSINESS DETECTED"
            self.alert_active = True
        elif distracted:
            self.safety_score -= config.DISTRACTION_PENALTY
            self.alert_message = "⚠ DISTRACTION DETECTED"
            self.alert_active = True
        else:
            # Regenerate score slowly during normal behavior
            self.safety_score += config.SCORE_REGENERATION
            self.alert_active = False
            self.alert_message = ""
        
        # Clamp score to valid range
        self.safety_score = max(
            config.MIN_SAFETY_SCORE,
            min(config.MAX_SAFETY_SCORE, self.safety_score)
        )
    
    def _determine_alert_color(self) -> Tuple[int, int, int]:
        """
        Determine UI color based on safety score.
        
        Returns:
            BGR color tuple
        """
        if self.safety_score <= config.CRITICAL_SCORE_THRESHOLD:
            return config.UI_COLOR_CRITICAL
        elif self.safety_score <= 60:
            return config.UI_COLOR_WARNING
        else:
            return config.UI_COLOR_NORMAL
    
    def _draw_dashboard(
        self,
        frame: np.ndarray,
        pitch: Optional[float],
        yaw: Optional[float],
        roll: Optional[float],
        ear: float,
        status: str
    ) -> np.ndarray:
        """
        Draw modern UI dashboard on frame.
        
        Displays:
        - Safety score with progress bar
        - Head pose angles
        - Eye Aspect Ratio
        - Alert status
        - System status
        
        Args:
            frame: Input frame
            pitch: Head pitch angle
            yaw: Head yaw angle
            roll: Head roll angle
            ear: Eye Aspect Ratio
            status: Current alertness status
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Top panel background
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Title
        cv2.putText(
            frame,
            "GuardianAI - Driver Monitor",
            (10, 30),
            config.UI_FONT,
            0.8,
            config.UI_COLOR_TEXT,
            2
        )
        
        # Safety Score with progress bar
        score_color = self._determine_alert_color()
        
        # Score label
        cv2.putText(
            frame,
            f"Safety Score: {int(self.safety_score)}",
            (10, 60),
            config.UI_FONT,
            0.7,
            config.UI_COLOR_TEXT,
            2
        )
        
        # Progress bar background
        bar_x, bar_y = 10, 70
        bar_width, bar_height = 300, 25
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
        # Progress bar fill
        fill_width = int((self.safety_score / config.MAX_SAFETY_SCORE) * bar_width)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            score_color,
            -1
        )
        
        # Progress bar border
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            config.UI_COLOR_TEXT,
            2
        )
        
        # Right side info panel
        info_x = w - 320
        
        # Head Pose
        if pitch is not None and yaw is not None and roll is not None:
            cv2.putText(
                frame,
                "Head Pose:",
                (info_x, 30),
                config.UI_FONT,
                0.6,
                config.UI_COLOR_TEXT,
                1
            )
            cv2.putText(
                frame,
                f"Pitch: {pitch:6.2f}°",
                (info_x, 55),
                config.UI_FONT,
                0.5,
                config.UI_COLOR_TEXT,
                1
            )
            cv2.putText(
                frame,
                f"Yaw:   {yaw:6.2f}°",
                (info_x, 75),
                config.UI_FONT,
                0.5,
                config.UI_COLOR_TEXT,
                1
            )
            cv2.putText(
                frame,
                f"Roll:  {roll:6.2f}°",
                (info_x, 95),
                config.UI_FONT,
                0.5,
                config.UI_COLOR_TEXT,
                1
            )
        
        # Eye Status (without EAR value)
        info_x2 = w - 160
        status_color = (
            config.UI_COLOR_NORMAL if status == "ALERT"
            else config.UI_COLOR_CRITICAL
        )
        cv2.putText(
            frame,
            f"Status: {status}",
            (info_x2, 40),
            config.UI_FONT,
            0.7,
            status_color,
            2
        )
        
        # Alert message (center bottom)
        if self.alert_active and self.alert_message:
            text_size = cv2.getTextSize(
                self.alert_message,
                config.UI_FONT,
                1.2,
                3
            )[0]
            
            text_x = (w - text_size[0]) // 2
            text_y = h - 50
            
            # Alert background
            padding = 20
            cv2.rectangle(
                overlay,
                (text_x - padding, text_y - text_size[1] - padding),
                (text_x + text_size[0] + padding, text_y + padding),
                (0, 0, 0),
                -1
            )
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
            
            # Alert text
            cv2.putText(
                frame,
                self.alert_message,
                (text_x, text_y),
                config.UI_FONT,
                1.2,
                config.UI_COLOR_CRITICAL,
                3
            )
        
        # Bottom status bar
        status_text = f"Frame: {self.frame_count} | Press 'q' to quit, 'r' to reset"
        cv2.putText(
            frame,
            status_text,
            (10, h - 10),
            config.UI_FONT,
            0.5,
            config.UI_COLOR_TEXT,
            1
        )
        
        return frame
    
    def _log_event_if_needed(
        self,
        event_type: str,
        pitch: Optional[float],
        yaw: Optional[float],
        roll: Optional[float],
        ear: float
    ) -> None:
        """
        Log events to file, avoiding duplicate consecutive logs.
        
        Args:
            event_type: Type of event
            pitch: Head pitch angle
            yaw: Head yaw angle
            roll: Head roll angle
            ear: Eye Aspect Ratio
        """
        # Log only when event changes or at intervals
        if event_type != self.last_event or self.frame_count % self.log_interval == 0:
            self.logger.log_event(
                event_type=event_type,
                score=self.safety_score,
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                ear=ear
            )
            self.last_event = event_type
            
            # Save snapshot for critical violations
            if self.safety_score <= config.CRITICAL_SCORE_THRESHOLD:
                self.logger.save_snapshot(
                    self.current_frame,
                    event_type,
                    self.safety_score
                )
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the monitoring pipeline.
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Annotated frame with dashboard
        """
        self.current_frame = frame.copy()
        self.frame_count += 1
        
        # 1. Estimate head pose
        pose_result, frame = self.head_pose_estimator.get_pose(frame)
        
        # 2. Detect drowsiness
        status, ear, frame = self.drowsiness_detector.detect_drowsiness(frame)
        
        # 3. Analyze results
        pitch, yaw, roll = (None, None, None)
        distracted = False
        drowsy = False
        
        if pose_result is not None:
            pitch, yaw, roll = pose_result
            
            # Check for distraction (excessive head rotation)
            if abs(yaw) > config.YAW_THRESHOLD:
                distracted = True
        
        if status == "DROWSY":
            drowsy = True
        
        # 4. Update safety score
        self._update_safety_score(drowsy, distracted)
        
        # 5. Determine event type for logging
        if drowsy:
            event_type = "DROWSY"
        elif distracted:
            event_type = "DISTRACTED"
        else:
            event_type = "NORMAL"
        
        # 6. Log event
        self._log_event_if_needed(event_type, pitch, yaw, roll, ear)
        
        # 7. Draw dashboard
        frame = self._draw_dashboard(frame, pitch, yaw, roll, ear, status)
        
        return frame
    
    def run(self) -> None:
        """
        Main application loop.
        
        Continuously captures and processes frames until user quits.
        Handles keyboard input for system control.
        """
        try:
            while True:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("✗ Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow("GuardianAI - Driver Monitoring System", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✓ Shutting down GuardianAI...")
                    break
                elif key == ord('r'):
                    self.safety_score = config.INITIAL_SAFETY_SCORE
                    self.drowsiness_detector.reset()
                    print("✓ Safety score reset to 100")
                elif key == ord('s'):
                    self.logger.save_snapshot(
                        self.current_frame,
                        "MANUAL",
                        self.safety_score
                    )
                    print("✓ Manual snapshot saved")
                elif key == ord('c'):
                    self.logger.clear_logs()
                    print("✓ Logs cleared")
        
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        
        except Exception as e:
            print(f"\n✗ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """
        Clean up resources and close windows.
        """
        print("\nCleaning up resources...")
        self.camera.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")
        print("=" * 60)


def main() -> None:
    """
    Application entry point.
    """
    try:
        system = GuardianSystem()
        system.run()
    except RuntimeError as e:
        print(f"\n✗ Failed to initialize system: {e}")
        print("Please check your camera connection and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()