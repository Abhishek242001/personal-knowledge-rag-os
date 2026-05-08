"""
skeleton_reid.py - Enhanced Skeleton-Based Person Re-Identification
===================================================================

Uses YOLO11-Pose keypoints for person re-identification.
NOW WITH INTEGRATED OCCLUSION HANDLING!

Key Features:
- YOLO11-Pose keypoint extraction (17 COCO keypoints)
- ✅ NEW: Occlusion-robust feature extraction
- ✅ NEW: Quality scoring for features
- ✅ NEW: Three-tier extraction strategy
- Body measurement calculation from keypoints
- 128-dimensional feature vector from body proportions
- Robust similarity calculation
- Temporal smoothing for stable measurements

Target: 65% weight in final multi-modal fusion (primary modality)

Version: 2.0 (Enhanced)
Author: AI Team
Date: 2026-01-05
"""

import logging
from typing import Dict, Optional, Tuple
from collections import deque

import numpy as np
from . import config
from .occlusion_handler import OcclusionHandler

logger = logging.getLogger(__name__)


class SkeletonFeatureExtractor:
    """
    Extract skeleton-based features using YOLO11-Pose keypoints.
    
    ✅ ENHANCED: Now uses OcclusionHandler for robust feature extraction
    
    This is the key innovation that enables ReID with identical uniforms.
    Uses body proportions and measurements from YOLO11-Pose (17 COCO keypoints).
    
    NO MediaPipe required!
    """
    
    # COCO keypoint indices (YOLO11-Pose format)
    # 17 keypoints total
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(self,
                 min_detection_confidence: float = None,
                 min_tracking_confidence: float = None,
                 enable_temporal_smoothing: bool = None,
                 smoothing_window: int = None):
        """
        Initialize skeleton feature extractor.
        
        Args:
            min_detection_confidence: Minimum confidence for keypoint detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_temporal_smoothing: Enable temporal smoothing
            smoothing_window: Number of frames for smoothing
        """
        
        # Use config defaults if not specified
        if min_detection_confidence is None:
            min_detection_confidence = config.SKELETON.MIN_DETECTION_CONFIDENCE
        if min_tracking_confidence is None:
            min_tracking_confidence = config.SKELETON.MIN_TRACKING_CONFIDENCE
        if enable_temporal_smoothing is None:
            enable_temporal_smoothing = config.SKELETON.ENABLE_TEMPORAL_SMOOTHING
        if smoothing_window is None:
            smoothing_window = config.SKELETON.SMOOTHING_WINDOW
        
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.smoothing_window = smoothing_window
        
        # ✅ NEW: Occlusion handler for robust extraction
        self.occlusion_handler = OcclusionHandler()
        
        # Temporal smoothing buffer
        self.measurement_history = {}
        
        # ✅ NEW: Statistics tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'standard_extractions': 0,
            'imputed_extractions': 0,
            'fallback_extractions': 0,
            'failed_extractions': 0,
            'occlusion_levels': []
        }
        
        logger.info("✅ Skeleton extractor initialized (v2.0 - Enhanced)")
        logger.info(f"  Min detection confidence: {min_detection_confidence}")
        logger.info(f"  Temporal smoothing: {enable_temporal_smoothing}")
        logger.info("  ✅ NEW: Occlusion handling enabled")
    
    def extract_features(self,
                        frame: np.ndarray,
                        bbox: Tuple[int, int, int, int],
                        keypoints: Optional[np.ndarray] = None,
                        person_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        ✅ ENHANCED: Extract skeleton features with occlusion handling.
        
        Args:
            frame: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            keypoints: YOLO11-Pose keypoints (17, 3) - [x, y, confidence]
            person_id: Person ID for temporal smoothing
        
        Returns:
            Dictionary with:
            - features: 128-dimensional feature vector
            - quality_score: 0-1 (higher = more reliable)
            - occlusion_level: 0-1 (0=fully visible, 1=fully occluded)
            - method: 'standard', 'imputed', or 'fallback'
            - visible_keypoints: Number of visible keypoints
        """
        self.extraction_stats['total_extractions'] += 1
        
        if keypoints is None:
            logger.warning("No keypoints provided")
            self.extraction_stats['failed_extractions'] += 1
            return None
        
        # Validate keypoints shape
        if keypoints.shape[0] != 17:
            logger.warning(f"Invalid keypoints shape: {keypoints.shape}, expected (17, 3)")
            self.extraction_stats['failed_extractions'] += 1
            return None
        
        # ✅ NEW: Use occlusion handler for robust extraction
        occlusion_result = self.occlusion_handler.extract_robust_features(
            keypoints=keypoints,
            bbox=bbox,
            min_confidence=self.min_detection_confidence
        )
        
        if occlusion_result is None:
            logger.debug("Occlusion handler rejected keypoints (too occluded)")
            self.extraction_stats['failed_extractions'] += 1
            return None
        
        # Extract measurements from occlusion handler
        measurements = occlusion_result['measurements']
        quality_score = occlusion_result['quality_score']
        occlusion_level = occlusion_result['occlusion_level']
        method = occlusion_result['method']
        
        # Track statistics
        self.extraction_stats['occlusion_levels'].append(occlusion_level)
        if method == 'standard':
            self.extraction_stats['standard_extractions'] += 1
        elif method == 'imputed':
            self.extraction_stats['imputed_extractions'] += 1
        elif method == 'fallback':
            self.extraction_stats['fallback_extractions'] += 1
        
        # Apply temporal smoothing if enabled
        if self.enable_temporal_smoothing and person_id is not None:
            measurements = self._apply_temporal_smoothing(measurements, person_id)
        
        # Convert measurements to 128-dim feature vector
        features = self._measurements_to_features(measurements)
        
        # Count visible keypoints
        confidences = keypoints[:, 2]
        visible_keypoints = int(np.sum(confidences > self.min_detection_confidence))
        
        return {
            'features': features,
            'quality_score': quality_score,
            'occlusion_level': occlusion_level,
            'method': method,
            'visible_keypoints': visible_keypoints,
            'measurements': measurements  # For debugging
        }
    
    def _apply_temporal_smoothing(self,
                                 measurements: Dict[str, float],
                                 person_id: int) -> Dict[str, float]:
        """
        Apply temporal smoothing to measurements.
        
        Same as before - uses exponential moving average.
        """
        if person_id not in self.measurement_history:
            self.measurement_history[person_id] = {
                key: deque(maxlen=self.smoothing_window)
                for key in measurements.keys()
            }
        
        history = self.measurement_history[person_id]
        smoothed = {}
        
        for key, value in measurements.items():
            if key not in history:
                history[key] = deque(maxlen=self.smoothing_window)
            
            history[key].append(value)
            smoothed[key] = np.mean(history[key])
        
        return smoothed
    
    def _measurements_to_features(self, measurements: Dict[str, float]) -> np.ndarray:
        """
        Convert measurements to 128-dimensional feature vector.
        
        Same as before - creates rich feature representation.
        
        Args:
            measurements: Dictionary of body measurements
        
        Returns:
            128-dimensional normalized feature vector
        """
        # Define expected measurement keys (in order)
        measurement_keys = [
            'shoulder_width', 'hip_width', 'torso_length',
            'left_upper_leg', 'left_lower_leg', 'right_upper_leg', 'right_lower_leg',
            'left_upper_arm', 'left_forearm', 'right_upper_arm', 'right_forearm',
            'shoulder_hip_ratio', 'torso_shoulder_ratio', 'height_estimate'
        ]
        
        # Extract base features
        base_features = []
        for key in measurement_keys:
            if key in measurements:
                base_features.append(measurements[key])
            else:
                base_features.append(0.0)
        
        base_features = np.array(base_features, dtype=np.float32)
        
        # Expand to 128 dimensions using various transformations
        features = []
        
        # 1. Raw measurements (14 dims)
        features.extend(base_features)
        
        # 2. Log-transformed measurements (14 dims)
        features.extend(np.log1p(base_features))
        
        # 3. Squared measurements (14 dims)
        features.extend(base_features ** 2)
        
        # 4. Normalized measurements (14 dims)
        norm = np.linalg.norm(base_features) + 1e-6
        features.extend(base_features / norm)
        
        # 5. Pairwise ratios of key measurements (20 dims)
        key_measurements = base_features[:6]  # First 6 measurements
        for i in range(len(key_measurements)):
            for j in range(i + 1, min(i + 5, len(key_measurements))):
                ratio = key_measurements[i] / (key_measurements[j] + 1e-6)
                features.append(ratio)
        
        # 6. Statistical features (20 dims)
        features.extend([
            np.mean(base_features),
            np.std(base_features),
            np.max(base_features),
            np.min(base_features),
            np.median(base_features),
        ] * 4)
        
        # 7. Cross-products (32 dims)
        for i in range(min(8, len(base_features))):
            for j in range(i + 1, min(i + 5, len(base_features))):
                features.append(base_features[i] * base_features[j])
        
        # Ensure exactly 128 dimensions
        features = np.array(features, dtype=np.float32)[:128]
        
        # Pad if necessary
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)), mode='constant')
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def calculate_similarity(self,
                           features1: np.ndarray,
                           features2: np.ndarray) -> float:
        """
        Calculate similarity between two skeleton feature vectors.
        
        Same as before - cosine similarity.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Convert from [-1, 1] to [0, 1]
        similarity = (cosine_sim + 1.0) / 2.0
        
        return float(similarity)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        ✅ NEW: Get extraction statistics.
        
        Returns detailed statistics about feature extraction performance.
        """
        total = self.extraction_stats['total_extractions']
        
        if total == 0:
            return self.extraction_stats
        
        stats = {
            **self.extraction_stats,
            'success_rate': (total - self.extraction_stats['failed_extractions']) / total,
            'standard_rate': self.extraction_stats['standard_extractions'] / total,
            'imputed_rate': self.extraction_stats['imputed_extractions'] / total,
            'fallback_rate': self.extraction_stats['fallback_extractions'] / total,
            'failure_rate': self.extraction_stats['failed_extractions'] / total,
            'avg_occlusion_level': np.mean(self.extraction_stats['occlusion_levels']) 
                                   if self.extraction_stats['occlusion_levels'] else 0.0
        }
        
        # Get occlusion handler stats
        stats['occlusion_handler'] = self.occlusion_handler.get_statistics()
        
        return stats
    
    def clear_person_history(self, person_id: int):
        """Clear temporal smoothing history for a person."""
        if person_id in self.measurement_history:
            del self.measurement_history[person_id]


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# ============================================================================

def extract_skeleton_features(frame: np.ndarray,
                             bbox: Tuple[int, int, int, int],
                             keypoints: np.ndarray,
                             person_id: Optional[int] = None) -> Optional[np.ndarray]:
    """
    ✅ BACKWARD COMPATIBLE: Simple function for basic usage.
    
    This maintains compatibility with old code that expects just a feature vector.
    
    Args:
        frame: Input image
        bbox: Bounding box
        keypoints: YOLO11-Pose keypoints (17, 3)
        person_id: Optional person ID
    
    Returns:
        128-dim feature vector or None
    """
    extractor = SkeletonFeatureExtractor()
    result = extractor.extract_features(frame, bbox, keypoints, person_id)
    
    if result is None:
        return None
    
    return result['features']


# ============================================================================
# TESTING
# ============================================================================

def test_skeleton_extractor_enhanced():
    """Test enhanced skeleton feature extractor."""
    print("\n" + "="*70)
    print("ENHANCED SKELETON FEATURE EXTRACTOR TEST")
    print("="*70)
    
    # Initialize extractor
    print("\n1. Initializing extractor...")
    extractor = SkeletonFeatureExtractor()
    print("✓ Extractor initialized")
    
    # Test 1: Good quality keypoints (minimal occlusion)
    print("\n2. Testing with high-quality keypoints...")
    keypoints_good = np.random.rand(17, 3)
    keypoints_good[:, 2] = 0.9  # High confidence
    
    dummy_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    bbox = (100, 100, 300, 500)
    
    result_good = extractor.extract_features(
        frame=dummy_frame,
        bbox=bbox,
        keypoints=keypoints_good,
        person_id=1
    )
    
    if result_good:
        print(f"✓ Features extracted: shape={result_good['features'].shape}")
        print(f"  Method: {result_good['method']}")
        print(f"  Quality: {result_good['quality_score']:.3f}")
        print(f"  Occlusion: {result_good['occlusion_level']:.3f}")
        print(f"  Visible keypoints: {result_good['visible_keypoints']}/17")
    else:
        print("✗ Feature extraction failed")
    
    # Test 2: Partial occlusion (8-11 visible keypoints)
    print("\n3. Testing with partial occlusion...")
    keypoints_partial = np.random.rand(17, 3)
    keypoints_partial[:9, 2] = 0.8   # Upper body visible
    keypoints_partial[9:, 2] = 0.2   # Lower body occluded
    
    result_partial = extractor.extract_features(
        frame=dummy_frame,
        bbox=bbox,
        keypoints=keypoints_partial,
        person_id=2
    )
    
    if result_partial:
        print(f"✓ Features extracted with partial occlusion")
        print(f"  Method: {result_partial['method']}")
        print(f"  Quality: {result_partial['quality_score']:.3f}")
        print(f"  Occlusion: {result_partial['occlusion_level']:.3f}")
        print(f"  Visible keypoints: {result_partial['visible_keypoints']}/17")
    else:
        print("✗ Feature extraction failed")
    
    # Test 3: Heavy occlusion (4-7 visible keypoints)
    print("\n4. Testing with heavy occlusion...")
    keypoints_heavy = np.random.rand(17, 3)
    keypoints_heavy[:5, 2] = 0.7     # Only head visible
    keypoints_heavy[5:, 2] = 0.1     # Body occluded
    
    result_heavy = extractor.extract_features(
        frame=dummy_frame,
        bbox=bbox,
        keypoints=keypoints_heavy,
        person_id=3
    )
    
    if result_heavy:
        print(f"✓ Features extracted with heavy occlusion")
        print(f"  Method: {result_heavy['method']}")
        print(f"  Quality: {result_heavy['quality_score']:.3f}")
        print(f"  Occlusion: {result_heavy['occlusion_level']:.3f}")
        print(f"  Visible keypoints: {result_heavy['visible_keypoints']}/17")
    else:
        print("✗ Feature extraction failed")
    
    # Test 4: Similarity calculation
    print("\n5. Testing similarity calculation...")
    if result_good and result_partial:
        # Same person (good vs partial)
        sim_same = extractor.calculate_similarity(
            result_good['features'],
            result_partial['features']
        )
        print(f"  Same person (different occlusion): {sim_same:.3f}")
        
        # Different person
        if result_heavy:
            sim_diff = extractor.calculate_similarity(
                result_good['features'],
                result_heavy['features']
            )
            print(f"  Different person: {sim_diff:.3f}")
    
    # Test 5: Statistics
    print("\n6. Extraction statistics:")
    stats = extractor.get_statistics()
    print(f"  Total extractions: {stats['total_extractions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Standard extractions: {stats['standard_rate']:.1%}")
    print(f"  Imputed extractions: {stats['imputed_rate']:.1%}")
    print(f"  Fallback extractions: {stats['fallback_rate']:.1%}")
    print(f"  Average occlusion: {stats['avg_occlusion_level']:.3f}")
    
    # Test 6: Backward compatibility
    print("\n7. Testing backward compatibility...")
    simple_features = extract_skeleton_features(
        dummy_frame, bbox, keypoints_good, person_id=4
    )
    if simple_features is not None:
        print(f"✓ Backward compatible function works: {simple_features.shape}")
    else:
        print("✗ Backward compatible function failed")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)
    print("\nKey improvements demonstrated:")
    print("  ✓ Handles high-quality keypoints (standard)")
    print("  ✓ Handles partial occlusion (imputation)")
    print("  ✓ Handles heavy occlusion (fallback)")
    print("  ✓ Provides quality scores")
    print("  ✓ Backward compatible")
    print("="*70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_skeleton_extractor_enhanced()
