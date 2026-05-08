"""
config.py - Enhanced System Configuration
=========================================

Central configuration for the Enhanced Multi-Camera Person ReID System v2.0

✅ NEW SECTIONS:
- ADAPTIVE_WEIGHTS: Adaptive weight adjustment parameters
- TEMPORAL: Temporal smoothing parameters
- OCCLUSION: Occlusion handling parameters
- HIERARCHICAL_MATCHING: Hierarchical matching parameters

Version: 2.0 (Enhanced)
Author: AI Team
Date: 2026-01-05
"""

from pathlib import Path


# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

class SYSTEM:
    """System-level configuration."""
    VERSION = "2.0"
    RELEASE_NAME = "Enhanced Multi-Camera ReID"
    ENABLE_ENHANCEMENTS = True  # Master switch for all enhancements


# ============================================================================
# INPUT/OUTPUT SETTINGS
# ============================================================================

class INPUT:
    """Input video configuration."""
    # Single video mode
    SINGLE_VIDEO = r"E:\UTC project\Person-Reid-Model - All Models\kichen_video\test3.mp4"
    
    # Multi-camera mode
    MULTI_CAMERA_MODE = False
    CAMERA_VIDEOS = {
        "cam1": "input/cam1.mp4",
        "cam2": "input/cam2.mp4",
        "cam3": "input/cam3.mp4"
    }


class OUTPUT:
    """Output configuration."""
    # Video output
    SAVE_VIDEO = True
    VIDEO_PATH = "output/annotated.mp4"
    DISPLAY_ANNOTATIONS = True
    
    # JSON output
    SAVE_JSON = True
    JSON_DIR = "../output/json"
    JSON_SUMMARY = True
    
    # Logging
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE = "output/reid_system.log"


# ============================================================================
# DETECTION SETTINGS (YOLO11-Pose)
# ============================================================================

class DETECTION:
    """YOLO11-Pose person detection settings."""
    MODEL_PATH = "yolo11n-pose.pt"  # YOLO11 nano pose model
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.7
    MAX_DETECTIONS = 30  # Maximum persons per frame
    
    # GPU optimization
    DEVICE = "cuda"  # "cuda" or "cpu"
    FP16 = True  # Half-precision for speed
    
    # Filtering
    MIN_BBOX_AREA = 2500  # Minimum 50x50 pixels (Fix #1)


# ============================================================================
# SKELETON REID SETTINGS
# ============================================================================

class SKELETON:
    """Skeleton-based ReID configuration."""
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Temporal smoothing
    ENABLE_TEMPORAL_SMOOTHING = True
    SMOOTHING_WINDOW = 5  # Number of frames to smooth
    
    # Feature extraction
    FEATURE_DIMENSION = 128


# ============================================================================
# ✅ NEW: OCCLUSION HANDLING
# ============================================================================

class OCCLUSION:
    """Occlusion handling configuration."""
    ENABLED = True
    
    # Keypoint thresholds for different strategies
    MIN_KEYPOINTS_STANDARD = 12   # ≥12: Standard extraction
    MIN_KEYPOINTS_PARTIAL = 8     # 8-11: Imputed extraction
    MIN_KEYPOINTS_HEAVY = 4       # 4-7: Fallback extraction
    # <4: Reject (too occluded)
    
    MIN_CONFIDENCE = 0.5
    
    # Bayesian imputation priors (body proportions)
    PRIORS = {
        'shoulder_width_height_ratio': (0.25, 0.03),      # mean, std
        'hip_shoulder_width_ratio': (0.85, 0.08),
        'torso_height_ratio': (0.30, 0.04),
        'upper_leg_height_ratio': (0.25, 0.03),
        'lower_leg_height_ratio': (0.25, 0.03),
        'upper_arm_shoulder_ratio': (0.45, 0.05),
        'forearm_upper_arm_ratio': (0.95, 0.10)
    }


# ============================================================================
# APPEARANCE REID SETTINGS (OSNet)
# ============================================================================

class APPEARANCE:
    """OSNet appearance-based ReID configuration."""
    MODEL_WEIGHTS = r"E:\UTC project\Person-Reid-Model - All Models\weight\reid-weights\osnet_ain_x1_0_imagenet.pth"
    FEATURE_DIMENSION = 512
    
    # Preprocessing
    INPUT_SIZE = (256, 128)  # Height x Width
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Filtering
    MIN_CROP_SIZE = (50, 100)  # Min height x width (Fix #1)
    
    # Quality thresholds
    MIN_ASPECT_RATIO = 1.5   # Prefer vertical crops
    MAX_ASPECT_RATIO = 3.0


# ============================================================================
# SPATIAL TRACKING SETTINGS
# ============================================================================

class SPATIAL:
    """Automatic spatial learning configuration."""
    # DBSCAN clustering parameters
    DBSCAN_EPS = 150.0           # Cluster radius in pixels
    DBSCAN_MIN_SAMPLES = 10      # Minimum points per cluster
    
    # Zone learning
    HISTORY_WINDOW = 300         # Positions to keep per person
    LEARNING_RATE = 0.1          # EMA learning rate for zone associations
    MIN_OBSERVATIONS = 20        # Min observations before using priors
    PRIMARY_ZONE_THRESHOLD = 0.6 # Threshold for primary zone assignment
    
    # Zone updates
    ZONE_UPDATE_INTERVAL = 100   # Frames between zone reclustering (Fix #3)
    
    # Multi-zone detection
    MULTI_ZONE_THRESHOLD = 3     # Zones visited to be considered multi-zone


# ============================================================================
# ✅ NEW: ADAPTIVE WEIGHTS
# ============================================================================

class ADAPTIVE_WEIGHTS:
    """Adaptive weight adjustment configuration."""
    ENABLED = True
    
    # Adaptation parameters
    ADAPTATION_RATE = 0.3        # How quickly weights adapt (0-1)
    HISTORY_WINDOW = 100         # Frames of history to track
    
    # Quality thresholds
    MIN_SKELETON_QUALITY = 0.3   # Below this: reduce skeleton weight
    MIN_APPEARANCE_QUALITY = 0.3 # Below this: reduce appearance weight
    MIN_SPATIAL_QUALITY = 0.3    # Below this: reduce spatial weight
    
    # Weight bounds
    MIN_WEIGHT = 0.05            # Minimum weight for any modality
    MAX_WEIGHT = 0.85            # Maximum weight for any modality


# ============================================================================
# ✅ NEW: TEMPORAL SMOOTHING
# ============================================================================

class TEMPORAL:
    """Temporal smoothing configuration."""
    ENABLED = True
    
    # Smoothing windows
    SIMILARITY_WINDOW = 5        # Frames for similarity smoothing
    FEATURE_WINDOW = 3           # Frames for feature smoothing
    POSITION_WINDOW = 10         # Frames for position smoothing
    
    # Smoothing parameters
    ALPHA = 0.7                  # EMA alpha (higher = more weight on current)
    
    # Feature weights (for weighted average)
    RECENT_WEIGHT = 1.0          # Weight for most recent frame
    WEIGHT_DECAY = 0.8           # Decay factor for older frames


# ============================================================================
# ✅ NEW: HIERARCHICAL MATCHING
# ============================================================================

class HIERARCHICAL_MATCHING:
    """Hierarchical matching configuration."""
    ENABLED = True
    
    # Pass 1: Skeleton-only
    SKELETON_ONLY_THRESHOLD = 0.85
    SKELETON_ONLY_ENABLED = True
    
    # Pass 2: Bimodal (skeleton + appearance)
    BIMODAL_THRESHOLD = 0.75
    BIMODAL_SKELETON_WEIGHT = 0.85
    BIMODAL_APPEARANCE_WEIGHT = 0.15
    BIMODAL_ENABLED = True
    
    # Pass 3: Full multimodal with adaptive threshold
    ADAPTIVE_THRESHOLD_MIN = 0.60
    ADAPTIVE_THRESHOLD_MAX = 0.80
    ADAPTIVE_THRESHOLD_BASE = 0.75
    
    # Threshold adjustments
    ZONE_CHANGE_PENALTY = -0.05        # Reduce threshold for zone changes
    MULTI_ZONE_PENALTY = -0.08         # Extra reduction for multi-zone persons
    STRONG_SKELETON_BONUS = -0.05      # Reduce threshold if skeleton >0.85
    TIME_GAP_PENALTY = -0.03           # Reduce threshold if time gap >5min
    LOW_QUALITY_PENALTY = -0.05        # Reduce threshold if low quality
    
    # Plausibility checks
    MAX_MOVEMENT_SPEED = 10.0          # Max pixels/second for plausibility
    TELEPORT_CHECK_WINDOW = 60.0       # Seconds window for teleport check


# ============================================================================
# FUSION SETTINGS
# ============================================================================

class FUSION:
    """Multi-modal fusion configuration."""
    # Base weights (can be overridden by adaptive weights)
    SKELETON_WEIGHT = 0.65       # Primary modality
    SPATIAL_WEIGHT = 0.25        # Secondary modality
    APPEARANCE_WEIGHT = 0.10     # Tertiary modality
    
    # Similarity thresholds
    SIMILARITY_THRESHOLD = 0.75  # Base threshold for matching
    
    # Zero embedding filter (Fix #2)
    ZERO_EMBEDDING_THRESHOLD = 1e-6


# ============================================================================
# GLOBAL DATABASE SETTINGS
# ============================================================================

class GLOBAL_DB:
    """Global ReID database configuration."""
    RETENTION_HOURS = 24.0       # Keep persons for 24 hours
    FEATURE_UPDATE_ALPHA = 0.3   # EMA for feature updates
    CLEANUP_INTERVAL = 100       # Frames between cleanup


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

class PERFORMANCE:
    """Performance optimization settings."""
    # GPU optimization
    USE_FP16 = True              # Half-precision for speed
    BATCH_SIZE = 1               # Single frame processing
    
    # Threading
    NUM_WORKERS = 4              # DataLoader workers
    PREFETCH = True              # Prefetch next frame
    
    # Memory management
    CLEAR_CACHE_INTERVAL = 500   # Frames between cache clearing


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """
    Validate configuration settings.
    
    Ensures all settings are within valid ranges and compatible.
    """
    errors = []
    warnings = []
    
    # Validate weights sum to 1.0
    weight_sum = FUSION.SKELETON_WEIGHT + FUSION.SPATIAL_WEIGHT + FUSION.APPEARANCE_WEIGHT
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"Fusion weights must sum to 1.0, got {weight_sum:.3f}")
    
    # Validate thresholds
    if not (0.0 <= FUSION.SIMILARITY_THRESHOLD <= 1.0):
        errors.append(f"FUSION.SIMILARITY_THRESHOLD must be in [0,1], got {FUSION.SIMILARITY_THRESHOLD}")
    
    if HIERARCHICAL_MATCHING.ENABLED:
        if HIERARCHICAL_MATCHING.SKELETON_ONLY_THRESHOLD < HIERARCHICAL_MATCHING.BIMODAL_THRESHOLD:
            warnings.append("Skeleton-only threshold should be >= bimodal threshold")
        
        if HIERARCHICAL_MATCHING.ADAPTIVE_THRESHOLD_MIN > HIERARCHICAL_MATCHING.ADAPTIVE_THRESHOLD_MAX:
            errors.append("Adaptive threshold min must be <= max")
    
    # Validate occlusion thresholds
    if OCCLUSION.ENABLED:
        if not (OCCLUSION.MIN_KEYPOINTS_HEAVY < OCCLUSION.MIN_KEYPOINTS_PARTIAL < OCCLUSION.MIN_KEYPOINTS_STANDARD):
            errors.append("Occlusion keypoint thresholds must be in ascending order")
    
    # Validate adaptive weights
    if ADAPTIVE_WEIGHTS.ENABLED:
        if not (0.0 < ADAPTIVE_WEIGHTS.ADAPTATION_RATE <= 1.0):
            errors.append(f"Adaptation rate must be in (0,1], got {ADAPTIVE_WEIGHTS.ADAPTATION_RATE}")
        
        if ADAPTIVE_WEIGHTS.MIN_WEIGHT >= ADAPTIVE_WEIGHTS.MAX_WEIGHT:
            errors.append("MIN_WEIGHT must be < MAX_WEIGHT")
    
    # Validate temporal smoothing
    if TEMPORAL.ENABLED:
        if not (0.0 < TEMPORAL.ALPHA <= 1.0):
            errors.append(f"Temporal alpha must be in (0,1], got {TEMPORAL.ALPHA}")
    
    # Print results
    if errors:
        print("❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration detected")
    
    if warnings:
        print("⚠️  CONFIGURATION WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ Configuration validation passed")


def print_config_summary():
    """Print configuration summary."""
    print("\n" + "="*70)
    print("ENHANCED MULTI-CAMERA REID SYSTEM v2.0 - CONFIGURATION")
    print("="*70)
    
    print("\n📹 INPUT/OUTPUT:")
    print(f"  Video: {INPUT.SINGLE_VIDEO}")
    print(f"  Save output: {OUTPUT.SAVE_VIDEO}")
    print(f"  JSON output: {OUTPUT.SAVE_JSON}")
    
    print("\n🔍 DETECTION:")
    print(f"  Model: {DETECTION.MODEL_PATH}")
    print(f"  Confidence: {DETECTION.CONFIDENCE_THRESHOLD}")
    print(f"  Device: {DETECTION.DEVICE}")
    print(f"  FP16: {DETECTION.FP16}")
    
    print("\n🦴 SKELETON REID:")
    print(f"  Min confidence: {SKELETON.MIN_DETECTION_CONFIDENCE}")
    print(f"  Feature dim: {SKELETON.FEATURE_DIMENSION}")
    print(f"  Temporal smoothing: {SKELETON.ENABLE_TEMPORAL_SMOOTHING}")
    
    print("\n🎭 APPEARANCE REID:")
    print(f"  Model: {APPEARANCE.MODEL_WEIGHTS}")
    print(f"  Feature dim: {APPEARANCE.FEATURE_DIMENSION}")
    print(f"  Input size: {APPEARANCE.INPUT_SIZE}")
    
    print("\n📍 SPATIAL TRACKING:")
    print(f"  DBSCAN eps: {SPATIAL.DBSCAN_EPS}px")
    print(f"  Min samples: {SPATIAL.DBSCAN_MIN_SAMPLES}")
    print(f"  Zone update interval: {SPATIAL.ZONE_UPDATE_INTERVAL} frames")
    
    print("\n🔀 FUSION:")
    print(f"  Skeleton weight: {FUSION.SKELETON_WEIGHT}")
    print(f"  Spatial weight: {FUSION.SPATIAL_WEIGHT}")
    print(f"  Appearance weight: {FUSION.APPEARANCE_WEIGHT}")
    print(f"  Similarity threshold: {FUSION.SIMILARITY_THRESHOLD}")
    
    if ADAPTIVE_WEIGHTS.ENABLED:
        print("\n✨ ADAPTIVE WEIGHTS:")
        print(f"  Adaptation rate: {ADAPTIVE_WEIGHTS.ADAPTATION_RATE}")
        print(f"  History window: {ADAPTIVE_WEIGHTS.HISTORY_WINDOW}")
        print(f"  Weight range: [{ADAPTIVE_WEIGHTS.MIN_WEIGHT}, {ADAPTIVE_WEIGHTS.MAX_WEIGHT}]")
    
    if TEMPORAL.ENABLED:
        print("\n⏱️  TEMPORAL SMOOTHING:")
        print(f"  Similarity window: {TEMPORAL.SIMILARITY_WINDOW}")
        print(f"  Feature window: {TEMPORAL.FEATURE_WINDOW}")
        print(f"  Position window: {TEMPORAL.POSITION_WINDOW}")
        print(f"  Alpha: {TEMPORAL.ALPHA}")
    
    if OCCLUSION.ENABLED:
        print("\n🔍 OCCLUSION HANDLING:")
        print(f"  Standard: ≥{OCCLUSION.MIN_KEYPOINTS_STANDARD} keypoints")
        print(f"  Imputed: {OCCLUSION.MIN_KEYPOINTS_PARTIAL}-{OCCLUSION.MIN_KEYPOINTS_STANDARD-1} keypoints")
        print(f"  Fallback: {OCCLUSION.MIN_KEYPOINTS_HEAVY}-{OCCLUSION.MIN_KEYPOINTS_PARTIAL-1} keypoints")
        print(f"  Reject: <{OCCLUSION.MIN_KEYPOINTS_HEAVY} keypoints")
    
    if HIERARCHICAL_MATCHING.ENABLED:
        print("\n🎯 HIERARCHICAL MATCHING:")
        print(f"  Pass 1 (Skeleton): {HIERARCHICAL_MATCHING.SKELETON_ONLY_THRESHOLD}")
        print(f"  Pass 2 (Bimodal): {HIERARCHICAL_MATCHING.BIMODAL_THRESHOLD}")
        print(f"  Pass 3 (Adaptive): {HIERARCHICAL_MATCHING.ADAPTIVE_THRESHOLD_MIN}-{HIERARCHICAL_MATCHING.ADAPTIVE_THRESHOLD_MAX}")
        print(f"  Zone change penalty: {HIERARCHICAL_MATCHING.ZONE_CHANGE_PENALTY}")
        print(f"  Multi-zone penalty: {HIERARCHICAL_MATCHING.MULTI_ZONE_PENALTY}")
    
    print("\n💾 GLOBAL DATABASE:")
    print(f"  Retention: {GLOBAL_DB.RETENTION_HOURS} hours")
    print(f"  Feature update alpha: {GLOBAL_DB.FEATURE_UPDATE_ALPHA}")
    print(f"  Cleanup interval: {GLOBAL_DB.CLEANUP_INTERVAL} frames")
    
    print("\n⚡ PERFORMANCE:")
    print(f"  FP16: {PERFORMANCE.USE_FP16}")
    print(f"  Batch size: {PERFORMANCE.BATCH_SIZE}")
    print(f"  Workers: {PERFORMANCE.NUM_WORKERS}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# RUN VALIDATION ON IMPORT
# ============================================================================

if __name__ == "__main__":
    validate_config()
    print_config_summary()
else:
    # Automatically validate when imported
    try:
        validate_config()
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        raise
