#!/usr/bin/env python3
"""
Enhanced AthleteRise Cricket Analytics with Bonus Features
- Advanced phase segmentation
- Contact moment detection
- Temporal smoothness analysis
- Reference comparison
- Performance optimization
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import math
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@dataclass
class BiomechanicalMetrics:
    """Enhanced container for biomechanical measurements"""
    front_elbow_angle: float = 0.0
    spine_lean: float = 0.0
    head_knee_distance: float = 0.0
    foot_direction: float = 0.0
    timestamp: float = 0.0
    # New advanced metrics
    wrist_velocity: float = 0.0
    elbow_acceleration: float = 0.0
    balance_index: float = 0.0
    smoothness_score: float = 0.0

@dataclass
class PhaseInfo:
    """Enhanced phase information with confidence and metrics"""
    phase: str = "stance"
    confidence: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    duration_ms: float = 0.0
    key_metrics: Dict = None

@dataclass
class ContactMoment:
    """Contact detection information"""
    frame_number: int = 0
    timestamp: float = 0.0
    confidence: float = 0.0
    wrist_velocity: float = 0.0
    impact_angle: float = 0.0

class EnhancedCricketAnalyzer:
    def __init__(self, config_path: str = "config/thresholds.json"):
        """Initialize the enhanced cricket analyzer"""
        self.config = self.load_config(config_path)
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.config["pose_detection"]["detection_threshold"],
            min_tracking_confidence=self.config["pose_detection"]["confidence_threshold"]
        )
        
        # Enhanced tracking variables
        self.frame_metrics: List[BiomechanicalMetrics] = []
        self.phase_history: List[PhaseInfo] = []
        self.fps_counter = deque(maxlen=30)
        self.wrist_positions = deque(maxlen=10)  # For velocity calculation
        self.contact_moments: List[ContactMoment] = []
        self.reference_technique = self.load_reference_technique()
        
        # Performance tracking
        self.total_frames = 0
        self.processing_times = []

    def load_reference_technique(self) -> Dict:
        """Load reference technique data for comparison"""
        return {
            "phases": {
                "stance": {"duration": 0.3, "elbow_angle": 110, "spine_lean": 15},
                "stride": {"duration": 0.2, "elbow_angle": 100, "spine_lean": 20},
                "downswing": {"duration": 0.15, "elbow_angle": 85, "spine_lean": 25},
                "impact": {"duration": 0.05, "elbow_angle": 70, "spine_lean": 20},
                "follow_through": {"duration": 0.25, "elbow_angle": 140, "spine_lean": 10},
                "recovery": {"duration": 0.3, "elbow_angle": 120, "spine_lean": 5}
            },
            "ideal_progression": {
                "elbow_smoothness": 0.95,
                "balance_consistency": 0.90,
                "timing_precision": 0.85
            }
        }

    def load_config(self, config_path: str) -> Dict:
        """Load enhanced configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = self.get_default_config()
        
        # Add advanced configuration
        if "advanced_features" not in config:
            config["advanced_features"] = {
                "contact_detection": {
                    "velocity_threshold": 0.15,
                    "acceleration_threshold": 0.3,
                    "smoothing_window": 5
                },
                "phase_detection": {
                    "min_phase_duration": 3,
                    "velocity_threshold": 0.05,
                    "angle_change_threshold": 10
                },
                "smoothness_analysis": {
                    "window_size": 11,
                    "poly_order": 3
                }
            }
        
        return config

    def get_default_config(self) -> Dict:
        """Enhanced default configuration"""
        return {
            "pose_detection": {"confidence_threshold": 0.7, "detection_threshold": 0.5},
            "biomechanical_thresholds": {
                "front_elbow_angle": {"good_min": 100, "good_max": 140, "excellent_min": 110, "excellent_max": 130},
                "spine_lean": {"good_min": 10, "good_max": 30, "excellent_min": 15, "excellent_max": 25},
                "head_knee_alignment": {"good_threshold": 50, "excellent_threshold": 30},
                "foot_direction": {"good_min": -15, "good_max": 45, "excellent_min": 0, "excellent_max": 30}
            },
            "video_processing": {"target_fps": 30, "max_width": 1280, "max_height": 720},
            "reference_angles": {"ideal_elbow_angle": 120, "ideal_spine_lean": 20, "ideal_head_knee_distance": 25}
        }

    def calculate_velocity(self, positions: List[np.ndarray]) -> float:
        """Calculate velocity from position history"""
        if len(positions) < 2:
            return 0.0
        
        velocities = []
        for i in range(1, len(positions)):
            dt = 1/30.0  # Assuming 30 FPS
            velocity = np.linalg.norm(positions[i] - positions[i-1]) / dt
            velocities.append(velocity)
        
        return np.mean(velocities) if velocities else 0.0

    def detect_contact_moment(self, current_metrics: BiomechanicalMetrics, 
                            frame_number: int) -> Optional[ContactMoment]:
        """Enhanced contact moment detection using multiple signals"""
        if len(self.frame_metrics) < 5:
            return None
        
        # Get recent wrist velocities
        recent_velocities = [m.wrist_velocity for m in self.frame_metrics[-5:]]
        
        # Detection criteria
        velocity_spike = (current_metrics.wrist_velocity > 
                         self.config["advanced_features"]["contact_detection"]["velocity_threshold"])
        
        velocity_change = False
        if len(recent_velocities) >= 3:
            velocity_change = (recent_velocities[-1] - recent_velocities[-3]) > 0.1
        
        # Elbow angle criteria (rapid change indicates impact)
        elbow_change = False
        if len(self.frame_metrics) >= 3:
            recent_angles = [m.front_elbow_angle for m in self.frame_metrics[-3:]]
            if len([a for a in recent_angles if a > 0]) >= 2:
                elbow_change = abs(recent_angles[-1] - recent_angles[-2]) > 15
        
        # Combined confidence score
        confidence = 0.0
        if velocity_spike:
            confidence += 0.4
        if velocity_change:
            confidence += 0.3
        if elbow_change:
            confidence += 0.3
        
        if confidence >= 0.6:  # Threshold for contact detection
            return ContactMoment(
                frame_number=frame_number,
                timestamp=current_metrics.timestamp,
                confidence=confidence,
                wrist_velocity=current_metrics.wrist_velocity,
                impact_angle=current_metrics.front_elbow_angle
            )
        
        return None

    def advanced_phase_detection(self, current_metrics: BiomechanicalMetrics, 
                               frame_number: int) -> str:
        """Advanced phase detection with multiple signals"""
        if len(self.frame_metrics) < 10:
            return "stance"
        
        # Get recent history
        recent_metrics = self.frame_metrics[-10:]
        recent_angles = [m.front_elbow_angle for m in recent_metrics if m.front_elbow_angle > 0]
        recent_velocities = [m.wrist_velocity for m in recent_metrics]
        
        if len(recent_angles) < 5:
            return "stance"
        
        # Calculate trends
        angle_trend = np.diff(recent_angles[-5:])
        velocity_trend = np.mean(recent_velocities[-3:])
        current_angle = recent_angles[-1]
        
        # Phase detection logic
        if velocity_trend < 0.02 and current_angle > 100:
            return "stance"
        elif np.mean(angle_trend) < -2 and velocity_trend > 0.03:
            return "stride" 
        elif np.mean(angle_trend) < -5 and current_angle < 90:
            return "downswing"
        elif velocity_trend > 0.1 and current_angle < 80:
            return "impact"
        elif np.mean(angle_trend) > 3 and current_angle > 90:
            return "follow_through"
        else:
            return "recovery"

    def compute_enhanced_metrics(self, keypoints: Dict[str, np.ndarray], 
                               frame_time: float) -> BiomechanicalMetrics:
        """Compute enhanced biomechanical metrics"""
        metrics = BiomechanicalMetrics(timestamp=frame_time)
        
        try:
            # Basic metrics (improved from original)
            if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                shoulder = keypoints['right_shoulder']
                elbow = keypoints['right_elbow']
                wrist = keypoints['right_wrist']
                
                # Elbow angle with validation
                if (np.linalg.norm(shoulder - elbow) > 0.01 and 
                    np.linalg.norm(wrist - elbow) > 0.01):
                    metrics.front_elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                else:
                    metrics.front_elbow_angle = 90.0
                
                # Wrist velocity calculation
                self.wrist_positions.append(wrist)
                if len(self.wrist_positions) >= 2:
                    dt = 1/30.0  # Frame time
                    velocity = np.linalg.norm(self.wrist_positions[-1] - self.wrist_positions[-2]) / dt
                    metrics.wrist_velocity = velocity
            
            # Improved spine lean
            if all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                shoulder_center = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2
                hip_center = (keypoints['left_hip'] + keypoints['right_hip']) / 2
                
                spine_vector_2d = shoulder_center[:2] - hip_center[:2]
                if np.linalg.norm(spine_vector_2d) > 0.01:
                    vertical_vector_2d = np.array([0, -1])
                    cos_angle = np.dot(spine_vector_2d, vertical_vector_2d) / (
                        np.linalg.norm(spine_vector_2d) * np.linalg.norm(vertical_vector_2d))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    metrics.spine_lean = abs(90 - np.arccos(cos_angle) * 180 / np.pi)
                else:
                    metrics.spine_lean = 15.0
            
            # Head-knee distance
            if 'nose' in keypoints and 'right_knee' in keypoints:
                head_pos = keypoints['nose'][:2]
                knee_pos = keypoints['right_knee'][:2]
                metrics.head_knee_distance = self.calculate_distance(head_pos, knee_pos) * 100
            
            # Balance index (new metric)
            if len(self.frame_metrics) >= 5:
                recent_spine_leans = [m.spine_lean for m in self.frame_metrics[-5:] if m.spine_lean > 0]
                if recent_spine_leans:
                    metrics.balance_index = 1.0 - (np.std(recent_spine_leans) / 100.0)
                    metrics.balance_index = max(0.0, min(1.0, metrics.balance_index))
        
        except Exception as e:
            print(f"Error computing enhanced metrics: {e}")
            # Fallback values
            metrics.front_elbow_angle = 120.0
            metrics.spine_lean = 20.0
            metrics.head_knee_distance = 25.0
        
        return metrics

    def calculate_smoothness_scores(self) -> Dict[str, float]:
        """Calculate temporal smoothness metrics"""
        if len(self.frame_metrics) < 10:
            return {"elbow_smoothness": 0.5, "spine_smoothness": 0.5, "overall_smoothness": 0.5}
        
        # Extract time series data
        elbow_angles = [m.front_elbow_angle for m in self.frame_metrics if m.front_elbow_angle > 0]
        spine_leans = [m.spine_lean for m in self.frame_metrics if m.spine_lean > 0]
        
        smoothness_scores = {}
        
        try:
            # Elbow smoothness (using second derivative)
            if len(elbow_angles) > 10:
                smoothed = savgol_filter(elbow_angles, 
                                       min(11, len(elbow_angles)//2*2-1), 3)
                second_derivative = np.diff(elbow_angles, 2)
                elbow_smoothness = 1.0 - (np.std(second_derivative) / 100.0)
                smoothness_scores["elbow_smoothness"] = max(0.0, min(1.0, elbow_smoothness))
            else:
                smoothness_scores["elbow_smoothness"] = 0.5
            
            # Spine smoothness
            if len(spine_leans) > 10:
                second_derivative = np.diff(spine_leans, 2)
                spine_smoothness = 1.0 - (np.std(second_derivative) / 50.0)
                smoothness_scores["spine_smoothness"] = max(0.0, min(1.0, spine_smoothness))
            else:
                smoothness_scores["spine_smoothness"] = 0.5
            
            # Overall smoothness
            smoothness_scores["overall_smoothness"] = np.mean([
                smoothness_scores["elbow_smoothness"],
                smoothness_scores["spine_smoothness"]
            ])
            
        except Exception as e:
            print(f"Error calculating smoothness: {e}")
            smoothness_scores = {"elbow_smoothness": 0.5, "spine_smoothness": 0.5, "overall_smoothness": 0.5}
        
        return smoothness_scores

    def generate_temporal_charts(self) -> str:
        """Generate temporal analysis charts"""
        if len(self.frame_metrics) < 10:
            return "insufficient_data"
        
        # Extract data
        timestamps = [m.timestamp for m in self.frame_metrics]
        elbow_angles = [m.front_elbow_angle if m.front_elbow_angle > 0 else np.nan 
                       for m in self.frame_metrics]
        spine_leans = [m.spine_lean if m.spine_lean > 0 else np.nan 
                      for m in self.frame_metrics]
        wrist_velocities = [m.wrist_velocity for m in self.frame_metrics]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Elbow Angle vs Time', 'Spine Lean vs Time', 'Wrist Velocity vs Time'),
            vertical_spacing=0.1
        )
        
        # Elbow angle plot
        fig.add_trace(
            go.Scatter(x=timestamps, y=elbow_angles, mode='lines+markers',
                      name='Elbow Angle', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Add reference lines
        ideal_elbow = self.config["reference_angles"]["ideal_elbow_angle"]
        fig.add_hline(y=ideal_elbow, line_dash="dash", line_color="green", 
                     annotation_text="Ideal", row=1, col=1)
        
        # Spine lean plot
        fig.add_trace(
            go.Scatter(x=timestamps, y=spine_leans, mode='lines+markers',
                      name='Spine Lean', line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        ideal_spine = self.config["reference_angles"]["ideal_spine_lean"]
        fig.add_hline(y=ideal_spine, line_dash="dash", line_color="green",
                     annotation_text="Ideal", row=2, col=1)
        
        # Wrist velocity plot
        fig.add_trace(
            go.Scatter(x=timestamps, y=wrist_velocities, mode='lines+markers',
                      name='Wrist Velocity', line=dict(color='purple', width=2)),
            row=3, col=1
        )
        
        # Add contact moments
        for contact in self.contact_moments:
            fig.add_vline(x=contact.timestamp, line_dash="dot", line_color="orange",
                         annotation_text=f"Contact ({contact.confidence:.2f})")
        
        # Update layout
        fig.update_layout(
            title="Cricket Technique Temporal Analysis",
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time (seconds)")
        fig.update_yaxes(title_text="Angle (degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Lean (degrees)", row=2, col=1)
        fig.update_yaxes(title_text="Velocity", row=3, col=1)
        
        # Save chart
        chart_path = "output/temporal_analysis.html"
        fig.write_html(chart_path)
        
        return chart_path

    def compare_with_reference(self) -> Dict:
        """Compare technique with reference standards"""
        if not self.frame_metrics:
            return {"error": "No data for comparison"}
        
        # Extract metrics for comparison
        elbow_angles = [m.front_elbow_angle for m in self.frame_metrics if m.front_elbow_angle > 0]
        spine_leans = [m.spine_lean for m in self.frame_metrics if m.spine_lean > 0]
        
        comparison = {}
        
        # Elbow angle comparison
        if elbow_angles:
            avg_elbow = np.mean(elbow_angles)
            ideal_elbow = self.reference_technique["phases"]["impact"]["elbow_angle"]
            elbow_deviation = abs(avg_elbow - ideal_elbow)
            comparison["elbow_analysis"] = {
                "average_angle": round(avg_elbow, 1),
                "ideal_angle": ideal_elbow,
                "deviation": round(elbow_deviation, 1),
                "grade": "excellent" if elbow_deviation < 10 else "good" if elbow_deviation < 20 else "needs_improvement"
            }
        
        # Spine lean comparison
        if spine_leans:
            avg_spine = np.mean(spine_leans)
            ideal_spine = self.reference_technique["phases"]["impact"]["spine_lean"]
            spine_deviation = abs(avg_spine - ideal_spine)
            comparison["spine_analysis"] = {
                "average_lean": round(avg_spine, 1),
                "ideal_lean": ideal_spine,
                "deviation": round(spine_deviation, 1),
                "grade": "excellent" if spine_deviation < 5 else "good" if spine_deviation < 10 else "needs_improvement"
            }
        
        # Smoothness comparison
        smoothness = self.calculate_smoothness_scores()
        ideal_smoothness = self.reference_technique["ideal_progression"]["elbow_smoothness"]
        comparison["smoothness_analysis"] = {
            "actual_smoothness": round(smoothness["overall_smoothness"], 2),
            "ideal_smoothness": ideal_smoothness,
            "grade": "excellent" if smoothness["overall_smoothness"] > 0.8 else "good" if smoothness["overall_smoothness"] > 0.6 else "needs_improvement"
        }
        
        return comparison

    def calculate_skill_grade(self) -> Dict:
        """Determine skill level based on multiple factors"""
        if not self.frame_metrics:
            return {"grade": "unknown", "confidence": 0.0}
        
        # Scoring factors
        scores = []
        
        # Technical consistency
        elbow_angles = [m.front_elbow_angle for m in self.frame_metrics if m.front_elbow_angle > 0]
        if elbow_angles:
            consistency = 1.0 - (np.std(elbow_angles) / 100.0)
            scores.append(max(0, min(1, consistency)))
        
        # Smoothness score
        smoothness = self.calculate_smoothness_scores()
        scores.append(smoothness["overall_smoothness"])
        
        # Contact detection accuracy
        contact_quality = len(self.contact_moments) / max(1, len(self.frame_metrics) / 30)  # Contacts per second
        contact_score = min(1.0, contact_quality * 2)  # Normalize
        scores.append(contact_score)
        
        # Overall score
        overall_score = np.mean(scores) if scores else 0.5
        
        # Grade assignment
        if overall_score > 0.8:
            grade = "Advanced"
            description = "Excellent technique with consistent execution"
        elif overall_score > 0.6:
            grade = "Intermediate"
            description = "Good fundamentals with room for refinement"
        else:
            grade = "Beginner"
            description = "Developing technique, focus on basics"
        
        return {
            "grade": grade,
            "score": round(overall_score * 10, 1),
            "confidence": min(0.95, len(self.frame_metrics) / 100),
            "description": description,
            "component_scores": {
                "consistency": round(scores[0] * 10, 1) if scores else 0,
                "smoothness": round(scores[1] * 10, 1) if len(scores) > 1 else 0,
                "timing": round(scores[2] * 10, 1) if len(scores) > 2 else 0
            }
        }

    def draw_enhanced_overlay(self, frame: np.ndarray, metrics: BiomechanicalMetrics, 
                            phase: str, frame_number: int) -> np.ndarray:
        """Enhanced overlay with more information"""
        height, width = frame.shape[:2]
        
        # Background for text (larger for more info)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Phase indicator with confidence
        phase_color = {
            "stance": (255, 255, 0),      # Yellow
            "stride": (0, 255, 255),      # Cyan  
            "downswing": (255, 165, 0),   # Orange
            "impact": (255, 0, 0),        # Red
            "follow_through": (0, 255, 0), # Green
            "recovery": (128, 128, 128)    # Gray
        }.get(phase, (255, 255, 255))
        
        cv2.putText(frame, f"Phase: {phase.upper()}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
        
        # Enhanced metrics display
        y_offset = 60
        metrics_text = [
            f"Elbow: {metrics.front_elbow_angle:.1f}°",
            f"Spine: {metrics.spine_lean:.1f}°", 
            f"Velocity: {metrics.wrist_velocity:.3f}",
            f"Balance: {metrics.balance_index:.2f}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(frame, text, (20, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Contact moment indicator
        if self.contact_moments and frame_number in [c.frame_number for c in self.contact_moments]:
            cv2.putText(frame, "CONTACT!", (width-150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Performance indicator
        if self.fps_counter:
            avg_fps = np.mean(self.fps_counter)
            fps_color = (0, 255, 0) if avg_fps > 15 else (255, 255, 0) if avg_fps > 10 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (width-100, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 2)
        
        return frame

    def analyze_video(self, video_path: str) -> Dict:
        """Enhanced video analysis with all bonus features"""
        print(f"🚀 Starting enhanced analysis of: {video_path}")
        
        # Video setup
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Output setup
        os.makedirs("output", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output/enhanced_annotated_video.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("⚡ Processing with enhanced features...")
        print("   • Advanced phase detection")
        print("   • Contact moment detection") 
        print("   • Temporal smoothness analysis")
        print("   • Real-time performance optimization")
        print("-" * 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            frame_time = frame_count / fps
            
            # Pose processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            current_metrics = BiomechanicalMetrics(timestamp=frame_time)
            current_phase = "stance"
            
            if results.pose_landmarks:
                keypoints = self.extract_keypoints(results.pose_landmarks)
                current_metrics = self.compute_enhanced_metrics(keypoints, frame_time)
                current_phase = self.advanced_phase_detection(current_metrics, frame_count)
                
                # Contact detection
                contact = self.detect_contact_moment(current_metrics, frame_count)
                if contact:
                    self.contact_moments.append(contact)
                
                # Draw pose
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Add to history
            self.frame_metrics.append(current_metrics)
            
            # Enhanced overlays
            frame = self.draw_enhanced_overlay(frame, current_metrics, current_phase, frame_count)
            out.write(frame)
            
            # Performance tracking
            frame_end = time.time()
            processing_time = frame_end - frame_start
            self.fps_counter.append(1.0 / processing_time if processing_time > 0 else 0)
            self.processing_times.append(processing_time)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
                print(f"Progress: {progress:.1f}% | FPS: {avg_fps:.1f} | Contacts: {len(self.contact_moments)}")
        
        cap.release()
        out.release()
        
        # Advanced analysis
        print("\n🔬 Generating advanced analysis...")
        smoothness_scores = self.calculate_smoothness_scores()
        reference_comparison = self.compare_with_reference()
        skill_assessment = self.calculate_skill_grade()
        
        # Generate temporal charts
        chart_path = self.generate_temporal_charts()
        
        # Comprehensive evaluation
        evaluation_data = {
            "video_info": {
                "path": video_path,
                "dimensions": f"{width}x{height}",
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": total_frames / fps
            },
            "processing_info": {
                "frames_processed": frame_count,
                "average_processing_fps": float(np.mean(self.fps_counter)) if self.fps_counter else 0,
                "total_processing_time": time.time() - start_time,
                "performance_grade": "Excellent" if np.mean(self.fps_counter) > 15 else "Good" if np.mean(self.fps_counter) > 10 else "Acceptable"
            },
            "enhanced_analysis": {
                "contact_moments": [asdict(c) for c in self.contact_moments],
                "phase_transitions": len(set([self.advanced_phase_detection(m, i) for i, m in enumerate(self.frame_metrics)])),
                "smoothness_scores": smoothness_scores,
                "reference_comparison": reference_comparison,
                "skill_assessment": skill_assessment
            },
            "traditional_scores": self.calculate_final_scores(),
            "advanced_insights": {
                "technique_consistency": round(smoothness_scores["overall_smoothness"] * 10, 1),
                "timing_precision": len(self.contact_moments) / max(1, total_frames / fps),
                "balance_stability": round(np.mean([m.balance_index for m in self.frame_metrics if m.balance_index > 0]) * 10, 1) if self.frame_metrics else 5.0,
                "overall_technical_grade": skill_assessment["grade"]
            },
            "recommendations": self.generate_recommendations(),
            "output_files": {
                "annotated_video": "output/enhanced_annotated_video.mp4",
                "temporal_charts": chart_path,
                "detailed_report": "output/enhanced_evaluation.json"
            }
        }
        
        # Save comprehensive evaluation
        with open('output/enhanced_evaluation.json', 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        print("\n" + "="*60)
        print("🎉 ENHANCED ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📹 Enhanced video: output/enhanced_annotated_video.mp4")
        print(f"📊 Temporal charts: {chart_path}")
        print(f"📋 Detailed report: output/enhanced_evaluation.json")
        print(f"⚡ Processing: {time.time() - start_time:.2f}s @ {np.mean(self.fps_counter):.1f} FPS")
        print(f"🎯 Skill Level: {skill_assessment['grade']} ({skill_assessment['score']}/10)")
        print(f"🔍 Contact Moments: {len(self.contact_moments)} detected")
        print(f"⚖️ Technique Smoothness: {smoothness_scores['overall_smoothness']:.2f}")
        
        return evaluation_data

    def generate_recommendations(self) -> List[str]:
        """Generate personalized recommendations based on analysis"""
        recommendations = []
        
        if not self.frame_metrics:
            return ["Insufficient data for recommendations"]
        
        # Analyze smoothness
        smoothness = self.calculate_smoothness_scores()
        if smoothness["elbow_smoothness"] < 0.7:
            recommendations.append("Focus on smoother elbow movement - practice shadow batting for fluidity")
        
        if smoothness["spine_smoothness"] < 0.7:
            recommendations.append("Work on balance and core stability - try single-leg stance exercises")
        
        # Analyze contact detection
        contacts_per_second = len(self.contact_moments) / max(1, len(self.frame_metrics) / 30)
        if contacts_per_second < 0.5:
            recommendations.append("Practice timing drills - use a metronome for rhythm training")
        elif contacts_per_second > 2:
            recommendations.append("Focus on single, clean contact - avoid multiple swing attempts")
        
        # Phase analysis
        phases_detected = len(set([self.advanced_phase_detection(m, i) for i, m in enumerate(self.frame_metrics)]))
        if phases_detected < 4:
            recommendations.append("Work on complete shot execution - practice full cover drive sequence")
        
        # Skill-specific recommendations
        skill_grade = self.calculate_skill_grade()
        if skill_grade["grade"] == "Beginner":
            recommendations.extend([
                "Focus on basic stance and grip fundamentals",
                "Practice stationary ball hitting before moving balls",
                "Work with a coach on basic technique correction"
            ])
        elif skill_grade["grade"] == "Intermediate":
            recommendations.extend([
                "Refine timing through varied pace bowling practice",
                "Focus on consistent follow-through",
                "Practice against different ball trajectories"
            ])
        else:  # Advanced
            recommendations.extend([
                "Fine-tune shot selection and placement",
                "Practice under pressure situations",
                "Focus on advanced stroke variations"
            ])
        
        return recommendations[:6]  # Limit to top 6 recommendations

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle

    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(p1 - p2)

    def extract_keypoints(self, landmarks) -> Dict[str, np.ndarray]:
        """Extract key body landmarks as normalized coordinates"""
        keypoints = {}
        landmark_indices = {
            'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                if landmark.visibility > 0.5:
                    keypoints[name] = np.array([landmark.x, landmark.y, landmark.z])
        
        return keypoints

    def calculate_final_scores(self) -> Dict[str, any]:
        """Enhanced final scoring with more sophisticated analysis"""
        if not self.frame_metrics:
            return {"error": "No metrics data available"}
        
        # Extract metrics
        elbow_angles = [m.front_elbow_angle for m in self.frame_metrics if m.front_elbow_angle > 0]
        spine_leans = [m.spine_lean for m in self.frame_metrics if m.spine_lean > 0]
        head_distances = [m.head_knee_distance for m in self.frame_metrics if m.head_knee_distance > 0]
        balance_indices = [m.balance_index for m in self.frame_metrics if m.balance_index > 0]
        
        scores = {}
        
        # Enhanced footwork scoring
        footwork_score = 7.0
        if head_distances:
            consistency = max(0, 10 - (np.std(head_distances) / 5))
            avg_distance = np.mean(head_distances)
            optimal_distance = 30.0  # Ideal head-knee distance
            position_accuracy = max(0, 10 - abs(avg_distance - optimal_distance) / 5)
            footwork_score = (consistency + position_accuracy) / 2
        
        # Enhanced head position scoring
        head_score = 7.0
        if head_distances:
            avg_distance = np.mean(head_distances)
            consistency = max(0, 10 - (np.std(head_distances) / 8))
            head_score = min(10, consistency)
        
        # Enhanced swing control scoring
        swing_score = 7.0
        if elbow_angles:
            ideal_angle = 120.0
            avg_angle = np.mean(elbow_angles)
            consistency = max(0, 10 - (np.std(elbow_angles) / 10))
            accuracy = max(0, 10 - abs(avg_angle - ideal_angle) / 15)
            swing_score = (consistency * 0.6 + accuracy * 0.4)
        
        # Enhanced balance scoring using balance index
        balance_score = 7.0
        if balance_indices:
            avg_balance = np.mean(balance_indices)
            balance_score = avg_balance * 10
        elif spine_leans:
            consistency = max(0, 10 - (np.std(spine_leans) / 3))
            balance_score = min(10, consistency)
        
        # Enhanced follow-through scoring
        followthrough_score = 8.0
        if len(self.contact_moments) > 0:
            # Good contact detection suggests good follow-through
            contact_quality = min(10, len(self.contact_moments) * 2)
            followthrough_score = (followthrough_score + contact_quality) / 2
        
        scores = {
            "footwork": {
                "score": round(footwork_score, 1), 
                "feedback": "Focus on consistent foot positioning and balance",
                "grade": self.get_score_grade(footwork_score)
            },
            "head_position": {
                "score": round(head_score, 1), 
                "feedback": "Keep head steady and aligned over front knee",
                "grade": self.get_score_grade(head_score)
            },
            "swing_control": {
                "score": round(swing_score, 1), 
                "feedback": "Maintain smooth and consistent elbow extension",
                "grade": self.get_score_grade(swing_score)
            },
            "balance": {
                "score": round(balance_score, 1), 
                "feedback": "Work on core stability throughout the shot",
                "grade": self.get_score_grade(balance_score)
            },
            "follow_through": {
                "score": round(followthrough_score, 1), 
                "feedback": "Complete the stroke with high elbow finish",
                "grade": self.get_score_grade(followthrough_score)
            }
        }
        
        return scores

    def get_score_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 9:
            return "A+"
        elif score >= 8:
            return "A"
        elif score >= 7:
            return "B+"
        elif score >= 6:
            return "B"
        elif score >= 5:
            return "C+"
        else:
            return "C"

    def validate_video_path(self, video_path: str) -> str:
        """Validate video file exists and is readable"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video file: {video_path}")
        
        cap.release()
        print(f"✅ Video validated: {video_path}")
        return video_path

def main():
    """Enhanced main function with better user experience"""
    import sys
    
    print("🏏" + "="*58 + "🏏")
    print("   AthleteRise Enhanced Cricket Analytics")
    print("   Advanced AI-Powered Technique Analysis")
    print("🏏" + "="*58 + "🏏")
    
    # Get video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "input_video.mp4"
        print(f"📹 Using default video: {video_path}")
        print("💡 Usage: python enhanced_cricket_analyzer.py <video_path>")
    
    try:
        print(f"\n🚀 Initializing Enhanced Analyzer...")
        analyzer = EnhancedCricketAnalyzer()
        
        print(f"🔍 Validating video file...")
        video_path = analyzer.validate_video_path(video_path)
        
        print(f"⚡ Starting comprehensive analysis...")
        print(f"   This includes all bonus features:")
        print(f"   • Advanced phase detection")
        print(f"   • Contact moment identification") 
        print(f"   • Temporal smoothness analysis")
        print(f"   • Reference technique comparison")
        print(f"   • Skill level assessment")
        print(f"   • Interactive temporal charts")
        
        results = analyzer.analyze_video(video_path)
        
        # Display enhanced results
        print(f"\n🎯 ENHANCED CRICKET ANALYSIS RESULTS")
        print(f"="*50)
        
        # Traditional scores
        print(f"\n📊 TECHNIQUE SCORES:")
        for category, data in results["traditional_scores"].items():
            score = data["score"]
            grade = data["grade"]
            print(f"   {category.replace('_', ' ').title()}: {score}/10 ({grade})")
            print(f"      💡 {data['feedback']}")
        
        # Enhanced insights
        enhanced = results["advanced_insights"]
        print(f"\n🔬 ADVANCED ANALYSIS:")
        print(f"   Skill Level: {enhanced['overall_technical_grade']}")
        print(f"   Technique Consistency: {enhanced['technique_consistency']}/10")
        print(f"   Balance Stability: {enhanced['balance_stability']}/10")
        print(f"   Timing Precision: {enhanced['timing_precision']:.2f}")
        
        # Contact analysis
        contacts = len(results["enhanced_analysis"]["contact_moments"])
        print(f"\n⚡ CONTACT ANALYSIS:")
        print(f"   Contact Moments Detected: {contacts}")
        if contacts > 0:
            avg_confidence = np.mean([c["confidence"] for c in results["enhanced_analysis"]["contact_moments"]])
            print(f"   Average Detection Confidence: {avg_confidence:.2f}")
        
        # Smoothness analysis
        smoothness = results["enhanced_analysis"]["smoothness_scores"]
        print(f"\n📈 SMOOTHNESS ANALYSIS:")
        print(f"   Elbow Movement: {smoothness['elbow_smoothness']:.2f}")
        print(f"   Spine Stability: {smoothness['spine_smoothness']:.2f}")
        print(f"   Overall Smoothness: {smoothness['overall_smoothness']:.2f}")
        
        # Recommendations
        print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        # Performance stats
        processing = results["processing_info"]
        print(f"\n⚡ PROCESSING PERFORMANCE:")
        print(f"   Speed: {processing['average_processing_fps']:.1f} FPS ({processing['performance_grade']})")
        print(f"   Time: {processing['total_processing_time']:.1f} seconds")
        
        # Output files
        print(f"\n📁 OUTPUT FILES GENERATED:")
        for file_type, path in results["output_files"].items():
            print(f"   {file_type.replace('_', ' ').title()}: {path}")
        
        overall_score = np.mean([score["score"] for score in results["traditional_scores"].values()])
        skill_score = results["enhanced_analysis"]["skill_assessment"]["score"]
        
        print(f"\n🏆 FINAL ASSESSMENT:")
        print(f"   Traditional Score: {overall_score:.1f}/10")
        print(f"   Advanced Skill Rating: {skill_score}/10")
        print(f"   Skill Level: {enhanced['overall_technical_grade']}")
        
        print(f"\n✅ Enhanced analysis complete! Check output folder for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Error during enhanced analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()