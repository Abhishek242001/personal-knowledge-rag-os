"""
main.py - Enhanced Multi-Camera Person Re-Identification System
================================================================

Complete person tracking and re-identification system with:
- YOLO11-Pose person detection
- Multi-modal ReID (skeleton + appearance + spatial)
- ✅ NEW: Hierarchical matching with adaptive thresholds
- ✅ NEW: Occlusion-robust feature extraction
- ✅ NEW: Temporal consistency and smoothing
- ✅ NEW: Adaptive weight adjustment
- ✅ NEW: Multi-zone person tracking
- Global cross-camera tracking
- GPU optimization
- JSON output

Version: 2.0 (Enhanced)
Author: AI Team
Date: 2026-01-05

ALL ENHANCEMENTS INTEGRATED:
✅ Hierarchical matching (global_reid_db.py)
✅ Occlusion handling (skeleton_reid.py + occlusion_handler.py)
✅ Adaptive weights (adaptive_weights.py)
✅ Temporal smoothing (temporal_smoother.py)
✅ Multi-zone tracking (spatial_tracker.py)
"""

import cv2
import numpy as np
import logging
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Any

# Import modules
from modules import config
from modules.person_detector import PersonDetector
from modules.skeleton_reid import SkeletonFeatureExtractor
from modules.osnet_reid import OSNetFeatureExtractor
from modules.global_reid_db import GlobalReIDDatabase
from modules.spatial_tracker import SpatialTracker
from modules.json_writer import JSONWriter

# ✅ NEW: Import enhanced modules
from modules.adaptive_weights import AdaptiveWeightManager
from modules.temporal_smoother import TemporalSmoother

logger = logging.getLogger(__name__)


class MultiCameraReIDSystem:
    """
    Enhanced multi-camera person re-identification system.
    
    ✅ NEW FEATURES (v2.0):
    1. Hierarchical matching (3-pass strategy)
    2. Adaptive weight adjustment (context-aware)
    3. Occlusion-robust features (intelligent imputation)
    4. Temporal smoothing (reduce jitter)
    5. Multi-zone person tracking (automatic detection)
    
    Features:
    - YOLO11-Pose detection
    - Skeleton + Appearance + Spatial features
    - Global person ID management
    - Cross-camera tracking
    - GPU optimization
    """
    
    def __init__(self, 
                 camera_id: str = "cam1",
                 enable_visualization: bool = True,
                 enable_json_output: bool = True,
                 enable_adaptive_weights: bool = True,
                 enable_temporal_smoothing: bool = True):
        """
        Initialize the enhanced ReID system.
        
        Args:
            camera_id: Unique camera identifier
            enable_visualization: Enable visualization overlay
            enable_json_output: Enable JSON file output
            enable_adaptive_weights: Enable adaptive weight adjustment
            enable_temporal_smoothing: Enable temporal smoothing
        """
        self.camera_id = camera_id
        self.enable_visualization = enable_visualization
        self.enable_json_output = enable_json_output
        self.enable_adaptive_weights = enable_adaptive_weights
        self.enable_temporal_smoothing = enable_temporal_smoothing
        
        logger.info("="*70)
        logger.info("ENHANCED MULTI-CAMERA REID SYSTEM v2.0")
        logger.info("="*70)
        logger.info(f"Initializing for camera: {camera_id}")
        
        # Initialize detector
        logger.info("\n[1/7] Loading person detector...")
        self.detector = PersonDetector(
            model_path=config.DETECTION.MODEL_PATH,
            confidence_threshold=config.DETECTION.CONFIDENCE_THRESHOLD
        )
        logger.info("✓ Person detector loaded")
        
        # Initialize ReID modules
        logger.info("\n[2/7] Loading ReID modules...")
        
        # Skeleton ReID (now with occlusion handling)
        self.skeleton_reid = SkeletonFeatureExtractor()
        logger.info("✓ Skeleton ReID loaded (with occlusion handling)")
        
        # Appearance ReID
        self.osnet = OSNetFeatureExtractor(
            weights_path=config.APPEARANCE.MODEL_WEIGHTS,
            use_fp16=config.PERFORMANCE.USE_FP16
        )
        logger.info("✓ OSNet loaded (GPU-optimized)")
        
        # Initialize enhanced global database
        logger.info("\n[3/7] Initializing enhanced global database...")
        self.global_db = GlobalReIDDatabase()
        logger.info("✓ Global database initialized (with hierarchical matching)")
        
        # Initialize spatial tracker
        logger.info("\n[4/7] Initializing spatial tracker...")
        self.spatial_tracker = SpatialTracker()
        logger.info("✓ Spatial tracker initialized (with multi-zone detection)")
        
        # ✅ NEW: Initialize adaptive weight manager
        if self.enable_adaptive_weights:
            logger.info("\n[5/7] Initializing adaptive weight manager...")
            self.adaptive_weights = AdaptiveWeightManager(
                adaptation_rate=0.3,
                history_window=100
            )
            logger.info("✓ Adaptive weights initialized")
        else:
            self.adaptive_weights = None
            logger.info("\n[5/7] Adaptive weights disabled")
        
        # ✅ NEW: Initialize temporal smoother
        if self.enable_temporal_smoothing:
            logger.info("\n[6/7] Initializing temporal smoother...")
            self.temporal_smoother = TemporalSmoother(
                similarity_window=5,
                feature_window=3,
                position_window=10,
                alpha=0.7
            )
            logger.info("✓ Temporal smoother initialized")
        else:
            self.temporal_smoother = None
            logger.info("\n[6/7] Temporal smoothing disabled")
        
        # Initialize JSON writer
        if enable_json_output:
            logger.info("\n[7/7] Initializing JSON writer...")
            self.json_writer = JSONWriter(
                output_dir=config.OUTPUT.JSON_DIR,
                create_summary=config.OUTPUT.JSON_SUMMARY
            )
            logger.info("✓ JSON writer initialized")
        else:
            self.json_writer = None
            logger.info("\n[7/7] JSON output disabled")
        
        # Performance tracking
        self.frame_count = 0
        self.last_frame_time = None
        self.fps_history = []
        
        # ✅ NEW: Enhanced statistics tracking
        self.stats = {
            'total_detections': 0,
            'skeleton_extractions': 0,
            'appearance_extractions': 0,
            'matches_found': 0,
            'new_persons_created': 0,
            'zone_changes': 0,
            'multi_zone_persons': 0,
            'adaptive_weight_adjustments': 0,
            'temporal_smoothing_applied': 0,
            'occlusion_imputations': 0
        }
        
        logger.info("\n" + "="*70)
        logger.info("✅ SYSTEM INITIALIZATION COMPLETE!")
        logger.info("="*70)
        logger.info("\nEnhancements active:")
        logger.info(f"  ✓ Hierarchical matching (3-pass strategy)")
        logger.info(f"  ✓ Occlusion-robust features (Bayesian imputation)")
        logger.info(f"  {'✓' if enable_adaptive_weights else '✗'} Adaptive weights (context-aware)")
        logger.info(f"  {'✓' if enable_temporal_smoothing else '✗'} Temporal smoothing (reduce jitter)")
        logger.info(f"  ✓ Multi-zone tracking (automatic detection)")
        logger.info("="*70 + "\n")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        ✅ ENHANCED: Process single frame through enhanced pipeline.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, response_dict)
        """
        self.frame_count += 1
        current_time = time.time()
        
        # ✅ ENHANCEMENT: Periodic zone learning (every N frames)
        if self.frame_count % config.SPATIAL.ZONE_UPDATE_INTERVAL == 0:
            self.spatial_tracker.update_zones()
            logger.debug(f"🔄 Spatial zones updated at frame {self.frame_count}")
        
        # Calculate FPS
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            if frame_time > 0:
                fps = 1.0 / frame_time
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
        self.last_frame_time = current_time
        
        # Calculate frame brightness (for adaptive weights)
        frame_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # Run detection
        detections = self.detector.detect(frame)
        self.stats['total_detections'] += len(detections)
        
        # Process each detection
        tracked_people = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            keypoints = detection.get('keypoints')
            
            # Calculate position
            x1, y1, x2, y2 = bbox
            position = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # ═══════════════════════════════════════════════════════════
            # ENHANCED FEATURE EXTRACTION
            # ═══════════════════════════════════════════════════════════
            
            # Extract skeleton features (now with occlusion handling)
            skeleton_result = None
            skeleton_features = None
            skeleton_quality = 0.5
            
            if keypoints is not None:
                skeleton_result = self.skeleton_reid.extract_features(
                    frame=frame,
                    bbox=bbox,
                    keypoints=keypoints,
                    person_id=None  # Will be assigned after matching
                )
                
                if skeleton_result:
                    skeleton_features = skeleton_result['features']
                    skeleton_quality = skeleton_result['quality_score']
                    
                    self.stats['skeleton_extractions'] += 1
                    
                    # Track occlusion imputations
                    if skeleton_result['method'] == 'imputed':
                        self.stats['occlusion_imputations'] += 1
                    
                    logger.debug(f"Skeleton: method={skeleton_result['method']}, "
                               f"quality={skeleton_quality:.2f}, "
                               f"occlusion={skeleton_result['occlusion_level']:.2f}")
            
            # Extract appearance features
            appearance_features = self.osnet.extract_features(frame, bbox)
            if appearance_features is not None:
                self.stats['appearance_extractions'] += 1
            
            # ✅ NEW: Apply temporal smoothing to features
            if self.enable_temporal_smoothing and self.temporal_smoother:
                if skeleton_features is not None:
                    # Note: We'll smooth after ID assignment
                    pass
                if appearance_features is not None:
                    # Note: We'll smooth after ID assignment  
                    pass
            
            # ✅ NEW: Calculate adaptive weights
            adaptive_weights = None
            if self.enable_adaptive_weights and self.adaptive_weights:
                adaptive_weights = self.adaptive_weights.calculate_adaptive_weights(
                    keypoints=keypoints,
                    crop_bbox=bbox,
                    position=position,
                    frame_brightness=frame_brightness,
                    distance_to_camera=None  # Could calculate if camera params available
                )
                self.stats['adaptive_weight_adjustments'] += 1
                
                logger.debug(f"Adaptive weights: {adaptive_weights}")
            
            # ═══════════════════════════════════════════════════════════
            # ENHANCED MATCHING WITH HIERARCHICAL STRATEGY
            # ═══════════════════════════════════════════════════════════
            
            # Get current zone from spatial tracker
            current_zone = self.spatial_tracker.get_zone_at_position(position)
            
            # Match with global database (now with hierarchical matching)
            match = self.global_db.find_match(
                camera_id=self.camera_id,
                skeleton_features=skeleton_features,
                appearance_features=appearance_features,
                position=position,
                current_zone=current_zone,
                spatial_tracker=self.spatial_tracker
            )
            
            if match:
                # ✅ ENHANCED: Now returns (person_id, similarity, match_info)
                person_id, similarity, match_info = match
                self.stats['matches_found'] += 1
                
                # Track zone changes
                if match_info.get('zone_change', False):
                    self.stats['zone_changes'] += 1
                
                # Log matching strategy used
                logger.debug(f"Match: ID={person_id}, sim={similarity:.3f}, "
                           f"strategy={match_info['strategy']}, "
                           f"threshold={match_info['threshold_used']:.2f}")
                
                # ✅ NEW: Apply temporal smoothing to features before updating
                if self.enable_temporal_smoothing and self.temporal_smoother:
                    if skeleton_features is not None:
                        skeleton_features = self.temporal_smoother.smooth_features(
                            person_id, skeleton_features
                        )
                        self.stats['temporal_smoothing_applied'] += 1
                    
                    if appearance_features is not None:
                        appearance_features = self.temporal_smoother.smooth_features(
                            person_id, appearance_features
                        )
                
                # Update existing person
                self.global_db.update_person(
                    person_id=person_id,
                    camera_id=self.camera_id,
                    skeleton_features=skeleton_features,
                    appearance_features=appearance_features,
                    position=position,
                    current_zone=current_zone
                )
                
            else:
                # Create new person
                person_id = self.global_db.add_person(
                    camera_id=self.camera_id,
                    skeleton_features=skeleton_features,
                    appearance_features=appearance_features,
                    position=position,
                    current_zone=current_zone
                )
                similarity = 1.0
                match_info = {'strategy': 'new_person', 'threshold_used': 0.0}
                
                self.stats['new_persons_created'] += 1
                logger.info(f"✨ New person created: ID={person_id}")
            
            # Update spatial tracker
            self.spatial_tracker.update_position(person_id, position)
            
            # Check if multi-zone person
            if self.spatial_tracker.is_multi_zone_person(person_id):
                if person_id not in [p['person_id'] for p in tracked_people 
                                    if p.get('is_multi_zone', False)]:
                    self.stats['multi_zone_persons'] += 1
            
            # ✅ NEW: Smooth position for visualization
            smooth_position = position
            if self.enable_temporal_smoothing and self.temporal_smoother:
                smooth_position = self.temporal_smoother.smooth_position(
                    person_id, position
                )
            
            # Prepare tracking info
            tracked_people.append({
                'person_id': person_id,
                'bbox': bbox,
                'confidence': confidence,
                'similarity': similarity,
                'position': position,
                'smooth_position': smooth_position,
                'has_keypoints': keypoints is not None,
                'current_zone': current_zone,
                'is_multi_zone': self.spatial_tracker.is_multi_zone_person(person_id),
                # ✅ NEW: Enhanced metadata
                'skeleton_quality': skeleton_quality if skeleton_result else 0.0,
                'skeleton_method': skeleton_result['method'] if skeleton_result else 'none',
                'match_strategy': match_info.get('strategy', 'unknown'),
                'match_threshold': match_info.get('threshold_used', 0.0),
                'zone_change': match_info.get('zone_change', False),
                'adaptive_weights': adaptive_weights
            })
        
        # Create response
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0.0
        
        response = {
            'frame_count': self.frame_count,
            'timestamp': current_time,
            'camera_id': self.camera_id,
            'fps': avg_fps,
            'total_people_detected': len(tracked_people),
            'detection': tracked_people,
            # ✅ NEW: Enhanced statistics
            'enhancements': {
                'adaptive_weights_enabled': self.enable_adaptive_weights,
                'temporal_smoothing_enabled': self.enable_temporal_smoothing,
                'zone_changes_detected': self.stats['zone_changes'],
                'multi_zone_persons': self.stats['multi_zone_persons'],
                'occlusion_imputations': self.stats['occlusion_imputations']
            }
        }
        
        # Write JSON if enabled
        if self.json_writer:
            self.json_writer.write_frame_response(self.frame_count, response)
        
        # Create visualization
        annotated_frame = self._draw_annotations(
            frame.copy(),
            tracked_people,
            avg_fps
        )
        
        return annotated_frame, response
    
    def _draw_annotations(self,
                         frame: np.ndarray,
                         tracked_people: List[Dict],
                         fps: float) -> np.ndarray:
        """
        ✅ ENHANCED: Draw annotations with new information.
        
        Args:
            frame: Input frame
            tracked_people: List of tracked person data
            fps: Current FPS
            
        Returns:
            Annotated frame
        """
        if not self.enable_visualization:
            return frame
        
        vis_frame = frame.copy()
        
        # Define colors for person IDs (cycling through palette)
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]
        
        for person in tracked_people:
            person_id = person['person_id']
            bbox = person['bbox']
            confidence = person['confidence']
            similarity = person['similarity']
            
            # Get color for this person ID
            color = colors[person_id % len(colors)]
            
            # ✅ NEW: Different color for multi-zone persons
            if person.get('is_multi_zone', False):
                color = (0, 165, 255)  # Orange - stands out
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # ✅ ENHANCED: More detailed label
            label_lines = [
                f"ID:{person_id}",
                f"C:{confidence:.2f}",
                f"S:{similarity:.2f}"
            ]
            
            # Add zone info if available
            if person.get('current_zone') is not None:
                zone_label = f"Z:{person['current_zone']}"
                if person.get('is_multi_zone', False):
                    zone_label += "*"  # Star for multi-zone
                label_lines.append(zone_label)
            
            # Add quality info
            if person.get('skeleton_quality', 0) > 0:
                qual = person['skeleton_quality']
                label_lines.append(f"Q:{qual:.2f}")
            
            # Combine into single label
            label = " ".join(label_lines)
            
            # Background for text
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_frame, (x1, y1 - label_h - 10), 
                (x1 + label_w, y1), color, -1
            )
            
            # Text
            cv2.putText(
                vis_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
            # ✅ NEW: Draw zone change indicator
            if person.get('zone_change', False):
                cv2.putText(
                    vis_frame, "ZONE CHANGE", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
                )
            
            # Draw center point (use smoothed position if available)
            pos = person.get('smooth_position', person['position'])
            cx, cy = map(int, pos)
            cv2.circle(vis_frame, (cx, cy), 4, color, -1)
        
        # ✅ ENHANCED: Draw system info
        info_lines = [
            f"Frame: {self.frame_count}",
            f"FPS: {fps:.1f}",
            f"People: {len(tracked_people)}",
            f"Camera: {self.camera_id}",
        ]
        
        # Add enhancement info
        if self.enable_adaptive_weights:
            info_lines.append("Adaptive: ON")
        if self.enable_temporal_smoothing:
            info_lines.append("Temporal: ON")
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                vis_frame, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 25
        
        return vis_frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """✅ ENHANCED: Get comprehensive system statistics."""
        stats = {
            'frame_count': self.frame_count,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0.0,
            'camera_id': self.camera_id,
            
            # Processing statistics
            'processing': {
                'total_detections': self.stats['total_detections'],
                'skeleton_extractions': self.stats['skeleton_extractions'],
                'appearance_extractions': self.stats['appearance_extractions'],
                'matches_found': self.stats['matches_found'],
                'new_persons_created': self.stats['new_persons_created'],
            },
            
            # ✅ NEW: Enhancement statistics
            'enhancements': {
                'zone_changes': self.stats['zone_changes'],
                'multi_zone_persons': self.stats['multi_zone_persons'],
                'adaptive_weight_adjustments': self.stats['adaptive_weight_adjustments'],
                'temporal_smoothing_applied': self.stats['temporal_smoothing_applied'],
                'occlusion_imputations': self.stats['occlusion_imputations'],
            },
            
            # Module statistics
            'global_db': self.global_db.get_statistics(),
            'spatial': self.spatial_tracker.get_statistics(),
            'skeleton_reid': self.skeleton_reid.get_statistics(),
            'osnet_filtered_crops': getattr(self.osnet, '_tiny_crops_filtered', 0),
        }
        
        # ✅ NEW: Add adaptive weights statistics
        if self.enable_adaptive_weights and self.adaptive_weights:
            stats['adaptive_weights'] = self.adaptive_weights.get_statistics()
        
        # ✅ NEW: Add temporal smoother statistics
        if self.enable_temporal_smoothing and self.temporal_smoother:
            stats['temporal_smoother'] = self.temporal_smoother.get_statistics()
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("\n" + "="*70)
        logger.info("CLEANING UP SYSTEM")
        logger.info("="*70)
        
        if self.json_writer:
            self.json_writer.close()
        
        # Print final statistics
        stats = self.get_statistics()
        
        logger.info("\n📊 FINAL STATISTICS:")
        logger.info(f"  Total Frames: {stats['frame_count']}")
        logger.info(f"  Average FPS: {stats['avg_fps']:.2f}")
        logger.info(f"  Total Detections: {stats['processing']['total_detections']}")
        logger.info(f"  Persons Tracked: {stats['global_db']['total_persons_tracked']}")
        logger.info(f"  Active Persons: {stats['global_db']['active_persons']}")
        
        logger.info("\n✨ ENHANCEMENTS PERFORMANCE:")
        logger.info(f"  Zone Changes Handled: {stats['enhancements']['zone_changes']}")
        logger.info(f"  Multi-Zone Persons: {stats['enhancements']['multi_zone_persons']}")
        logger.info(f"  Occlusion Imputations: {stats['enhancements']['occlusion_imputations']}")
        logger.info(f"  Adaptive Weight Adjustments: {stats['enhancements']['adaptive_weight_adjustments']}")
        logger.info(f"  Temporal Smoothing Applied: {stats['enhancements']['temporal_smoothing_applied']}")
        
        logger.info("\n🎯 MATCHING PERFORMANCE:")
        db_stats = stats['global_db']['matching_stats']
        logger.info(f"  Skeleton-Only Matches: {db_stats['skeleton_only_matches']}")
        logger.info(f"  Bimodal Matches: {db_stats['bimodal_matches']}")
        logger.info(f"  Full Multimodal Matches: {db_stats['full_multimodal_matches']}")
        logger.info(f"  Zone Change Matches: {db_stats['zone_change_matches']}")
        logger.info(f"  Threshold Adaptations: {db_stats['threshold_adaptations']}")
        
        logger.info("\n🗺️  SPATIAL TRACKING:")
        logger.info(f"  Zones Discovered: {stats['spatial']['num_zones']}")
        logger.info(f"  Zone Transitions: {stats['spatial']['total_transitions']}")
        logger.info(f"  Multi-Zone Persons: {stats['spatial']['multi_zone_persons']}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ CLEANUP COMPLETE")
        logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Camera Person ReID System v2.0"
    )
    parser.add_argument('--video', type=str, help='Input video path (overrides config)')
    parser.add_argument('--camera-id', type=str, default='cam1', help='Camera ID')
    parser.add_argument('--output', type=str, default='output/video.mp4', help='Output video path')
    parser.add_argument('--display', action='store_true', help='Display video')
    parser.add_argument('--no-json', action='store_true', help='Disable JSON output')
    parser.add_argument('--no-adaptive-weights', action='store_true', 
                       help='Disable adaptive weight adjustment')
    parser.add_argument('--no-temporal-smoothing', action='store_true',
                       help='Disable temporal smoothing')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.OUTPUT.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("\n" + "="*70)
    logger.info("ENHANCED MULTI-CAMERA PERSON RE-IDENTIFICATION SYSTEM v2.0")
    logger.info("="*70)
    logger.info("\n🚀 ALL ENHANCEMENTS ACTIVE:")
    logger.info("  ✓ Hierarchical Matching (3-pass strategy)")
    logger.info("  ✓ Occlusion-Robust Features (Bayesian imputation)")
    logger.info(f"  {'✓' if not args.no_adaptive_weights else '✗'} Adaptive Weights (context-aware)")
    logger.info(f"  {'✓' if not args.no_temporal_smoothing else '✗'} Temporal Smoothing (reduce jitter)")
    logger.info("  ✓ Multi-Zone Tracking (automatic detection)")
    logger.info("="*70 + "\n")
    
    # Initialize system
    system = MultiCameraReIDSystem(
        camera_id=args.camera_id,
        enable_visualization=True,
        enable_json_output=not args.no_json,
        enable_adaptive_weights=not args.no_adaptive_weights,
        enable_temporal_smoothing=not args.no_temporal_smoothing
    )
    
    # Get video path (command line overrides config)
    video_path = args.video if args.video else config.INPUT.SINGLE_VIDEO
    
    # Open video
    logger.info(f"📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Total Frames: {total_frames}")
    logger.info(f"  Duration: {total_frames/fps:.1f}s\n")
    
    # Setup output video
    if config.OUTPUT.SAVE_VIDEO:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logger.info(f"💾 Output video: {output_path}")
    else:
        out = None
    
    logger.info("🎬 Processing frames... (Press 'q' to quit)\n")
    
    try:
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Process frame
            annotated_frame, response = system.process_frame(frame)
            
            # Write output
            if out:
                out.write(annotated_frame)
            
            # Display
            if args.display or config.OUTPUT.DISPLAY_ANNOTATIONS:
                cv2.imshow('Enhanced Multi-Camera ReID v2.0', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("\n⚠️  User requested quit")
                    break
            
            # Progress
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                
                logger.info(f"Progress: {frame_idx}/{total_frames} ({progress:.1f}%) "
                          f"| FPS: {fps_actual:.1f} | People: {response['total_people_detected']}")
        
        # Final statistics
        elapsed = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("🎉 PROCESSING COMPLETE")
        logger.info("="*70)
        logger.info(f"Total frames processed: {frame_idx}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Average FPS: {frame_idx / elapsed:.2f}")
        
        stats = system.get_statistics()
        logger.info(f"\nPeople tracked: {stats['global_db']['total_persons_tracked']}")
        logger.info(f"Active persons: {stats['global_db']['active_persons']}")
        logger.info(f"Spatial zones discovered: {stats['spatial']['num_zones']}")
        logger.info(f"Multi-zone persons: {stats['spatial']['multi_zone_persons']}")
        logger.info(f"Occlusion imputations: {stats['enhancements']['occlusion_imputations']}")
        logger.info(f"Zone changes handled: {stats['enhancements']['zone_changes']}")
        logger.info("="*70 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        system.cleanup()
        
        if config.OUTPUT.SAVE_VIDEO:
            logger.info(f"\n✅ Output saved to: {output_path}")


if __name__ == "__main__":
    main()
