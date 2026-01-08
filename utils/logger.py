"""
Incident Logging Module
=======================
Handles logging of safety violations and evidence collection.

This module provides functionality to:
- Log timestamped events to CSV
- Save frame snapshots when critical violations occur
- Maintain an audit trail for safety analysis
"""

import cv2
import os
import csv
from datetime import datetime
from typing import Optional
import numpy as np
import config


class IncidentLogger:
    """
    Logs safety incidents and saves evidence snapshots.
    
    Maintains a CSV log file with timestamps and scores,
    and saves image snapshots for critical violations.
    """
    
    def __init__(self) -> None:
        """
        Initialize the logger and create necessary directories.
        """
        self.log_file = config.LOG_FILE
        self.evidence_folder = config.EVIDENCE_FOLDER
        
        # Create evidence folder if it doesn't exist
        if not os.path.exists(self.evidence_folder):
            os.makedirs(self.evidence_folder)
            print(f"✓ Created evidence folder: {self.evidence_folder}/")
        
        # Initialize CSV log file if it doesn't exist
        self._initialize_log_file()
        
    def _initialize_log_file(self) -> None:
        """
        Create CSV log file with headers if it doesn't exist.
        """
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Timestamp',
                        'Event_Type',
                        'Score',
                        'Pitch',
                        'Yaw',
                        'Roll',
                        'EAR',
                        'Severity'
                    ])
                print(f"✓ Created log file: {self.log_file}")
            except IOError as e:
                print(f"✗ Error creating log file: {e}")
    
    def log_event(
        self,
        event_type: str,
        score: float,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        roll: Optional[float] = None,
        ear: Optional[float] = None
    ) -> bool:
        """
        Log a safety event to the CSV file.
        
        Args:
            event_type: Type of event (e.g., "DROWSY", "DISTRACTED", "NORMAL")
            score: Current safety score (0-100)
            pitch: Head pitch angle (optional)
            yaw: Head yaw angle (optional)
            roll: Head roll angle (optional)
            ear: Eye Aspect Ratio (optional)
            
        Returns:
            True if logging successful, False otherwise
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine severity level
        severity = self._calculate_severity(score)
        
        # Format optional values
        pitch_str = f"{pitch:.2f}" if pitch is not None else "N/A"
        yaw_str = f"{yaw:.2f}" if yaw is not None else "N/A"
        roll_str = f"{roll:.2f}" if roll is not None else "N/A"
        ear_str = f"{ear:.4f}" if ear is not None else "N/A"
        
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    event_type,
                    f"{score:.2f}",
                    pitch_str,
                    yaw_str,
                    roll_str,
                    ear_str,
                    severity
                ])
            return True
        except IOError as e:
            print(f"✗ Error logging event: {e}")
            return False
    
    def _calculate_severity(self, score: float) -> str:
        """
        Calculate severity level based on safety score.
        
        Args:
            score: Current safety score (0-100)
            
        Returns:
            Severity string: "CRITICAL", "HIGH", "MEDIUM", or "LOW"
        """
        if score <= 30:
            return "CRITICAL"
        elif score <= 50:
            return "HIGH"
        elif score <= 70:
            return "MEDIUM"
        else:
            return "LOW"
    
    def save_snapshot(
        self,
        frame: np.ndarray,
        event_type: str,
        score: float
    ) -> Optional[str]:
        """
        Save a frame snapshot when a critical violation occurs.
        
        Snapshots are saved with timestamps and event information
        in the filename for easy identification and retrieval.
        
        Args:
            frame: Image frame to save (BGR format)
            event_type: Type of violation
            score: Current safety score
            
        Returns:
            Path to saved snapshot if successful, None otherwise
        """
        if frame is None or frame.size == 0:
            print("✗ Cannot save empty frame")
            return None
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.SNAPSHOT_PREFIX}_{event_type}_{timestamp}_score{int(score)}.jpg"
        filepath = os.path.join(self.evidence_folder, filename)
        
        try:
            # Add annotation to frame before saving
            annotated_frame = frame.copy()
            
            # Add timestamp overlay
            cv2.putText(
                annotated_frame,
                timestamp,
                (10, 30),
                config.UI_FONT,
                0.7,
                config.UI_COLOR_TEXT,
                2
            )
            
            # Add event type overlay
            cv2.putText(
                annotated_frame,
                f"Event: {event_type}",
                (10, 60),
                config.UI_FONT,
                0.7,
                config.UI_COLOR_CRITICAL,
                2
            )
            
            # Add score overlay
            cv2.putText(
                annotated_frame,
                f"Score: {int(score)}",
                (10, 90),
                config.UI_FONT,
                0.7,
                config.UI_COLOR_CRITICAL,
                2
            )
            
            # Save the frame
            success = cv2.imwrite(filepath, annotated_frame)
            
            if success:
                print(f"✓ Snapshot saved: {filepath}")
                return filepath
            else:
                print(f"✗ Failed to save snapshot: {filepath}")
                return None
                
        except Exception as e:
            print(f"✗ Error saving snapshot: {e}")
            return None
    
    def get_log_summary(self, last_n_entries: int = 10) -> list:
        """
        Retrieve the last N entries from the log file.
        
        Useful for displaying recent activity or generating reports.
        
        Args:
            last_n_entries: Number of recent entries to retrieve
            
        Returns:
            List of log entries (as dictionaries)
        """
        if not os.path.exists(self.log_file):
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                entries = list(reader)
                return entries[-last_n_entries:] if entries else []
        except IOError as e:
            print(f"✗ Error reading log file: {e}")
            return []
    
    def clear_logs(self) -> bool:
        """
        Clear all log entries (reinitialize the log file).
        
        Use with caution - this permanently deletes log history.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            self._initialize_log_file()
            print(f"✓ Log file cleared and reinitialized")
            return True
        except Exception as e:
            print(f"✗ Error clearing logs: {e}")
            return False