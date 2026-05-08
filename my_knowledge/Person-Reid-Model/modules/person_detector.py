"""
person_detector.py - YOLO11-Pose Person Detection Module (FIXED)
=================================================================

YOLO11-Pose detection with KEYPOINTS extraction.

✅ FIXED: Now extracts 17 COCO keypoints for skeleton features!

Author: AI Team
Date: 2024-12-29 (Fixed: 2025-01-03)
"""

import logging
from typing import List, Dict, Any
import sys
from pathlib import Path

import numpy as np

# Import config from root directory
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from modules import config

logger = logging.getLogger(__name__)

# Import YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available - YOLO detection disabled")


class PersonDetector:
    """
    YOLO11-Pose person detector.

    Features:
    - YOLO11-Pose person detection
    - 17 COCO keypoints extraction
    - Confidence threshold filtering
    - Bounding box extraction
    """

    def __init__(self, model_path: str, confidence_threshold: float = None):
        """
        Initialize person detector.

        Args:
            model_path: Path to YOLO model file (.pt)
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "Ultralytics not installed. Install with: pip install ultralytics"
            )

        # Use config default if not specified
        if confidence_threshold is None:
            confidence_threshold = config.DETECTION.CONFIDENCE_THRESHOLD

        self.confidence_threshold = confidence_threshold

        try:
            logger.info(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(model_path)

            # Force GPU if configured
            if config.PERFORMANCE.USE_GPU:
                import torch
                if torch.cuda.is_available():
                    # Note: YOLO handles .to('cuda') internally, just verify
                    logger.info(f"  Device: cuda (GPU mode enabled)")
                else:
                    logger.warning(f"  GPU requested but CUDA not available, using CPU")
            else:
                logger.info(f"  Device: cpu (GPU disabled in config)")
            logger.info(f"[OK] YOLO model loaded successfully")
            logger.info(f"  Confidence threshold: {confidence_threshold}")
            logger.info(f"  Model type: YOLO11-Pose (with keypoints)")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect people in frame with keypoints.

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: detection confidence score (0.0-1.0)
            - keypoints: (17, 3) array of [x, y, confidence] for 17 COCO keypoints
        """
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)

            detections = []

            # Process results
            for result in results:
                boxes = result.boxes

                if boxes is None:
                    continue

                # ✅ FIXED: Extract keypoints if available
                keypoints = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints

                for i, box in enumerate(boxes):
                    # Get confidence score
                    conf = float(box.conf[0])

                    # Filter by confidence threshold
                    if conf < self.confidence_threshold:
                        continue

                    # Get bounding box coordinates [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, xyxy)

                    # ✅ FIXED: Extract keypoints for this detection
                    person_keypoints = None
                    if keypoints is not None:
                        try:
                            # Get keypoints for this person
                            # YOLO11-Pose returns (num_people, 17, 3) tensor
                            kpts_data = keypoints.data[i].cpu().numpy()  # (17, 3)

                            if kpts_data.shape == (17, 3):
                                person_keypoints = kpts_data
                                logger.debug(f"Extracted {17} keypoints for detection {i}")
                            else:
                                logger.warning(f"Unexpected keypoints shape: {kpts_data.shape}")
                        except Exception as e:
                            logger.warning(f"Failed to extract keypoints: {e}")

                    # Add detection with keypoints
                    detection = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "keypoints": person_keypoints  # ✅ FIXED: Now includes keypoints!
                    }

                    detections.append(detection)

            # Log detection summary
            detections_with_kpts = sum(1 for d in detections if d['keypoints'] is not None)
            logger.debug(f"Detected {len(detections)} people "
                        f"({detections_with_kpts} with keypoints, threshold: {self.confidence_threshold})")

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return []

    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold.

        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")

        old_threshold = self.confidence_threshold
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated: {old_threshold} → {threshold}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            return {
                "model_type": "YOLO11-Pose",
                "confidence_threshold": self.confidence_threshold,
                "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
                "has_keypoints": True
            }
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {}
