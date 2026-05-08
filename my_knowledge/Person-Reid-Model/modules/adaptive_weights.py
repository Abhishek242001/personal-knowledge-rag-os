"""
adaptive_weights.py - Dynamic Weight Adjustment System
=====================================================

Adapts fusion weights based on real-time quality indicators.
Estimated accuracy improvement: +3-5%

Author: AI Team
Date: 2026-01-05
Version: 2.0
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModalityQuality:
    """Quality metrics for each modality."""
    skeleton_quality: float = 0.0
    appearance_quality: float = 0.0
    spatial_quality: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'skeleton': self.skeleton_quality,
            'appearance': self.appearance_quality,
            'spatial': self.spatial_quality
        }


class AdaptiveWeightManager:
    """
    Dynamically adjusts fusion weights based on:
    1. Keypoint visibility (skeleton quality)
    2. Crop size and lighting (appearance quality)
    3. Movement stability (spatial quality)
    4. Historical modality performance
    """
    
    def __init__(self,
                 base_skeleton_weight: float = 0.65,
                 base_spatial_weight: float = 0.25,
                 base_appearance_weight: float = 0.10,
                 adaptation_rate: float = 0.3,
                 history_window: int = 100):
        """
        Initialize adaptive weight manager.
        
        Args:
            base_skeleton_weight: Default skeleton weight
            base_spatial_weight: Default spatial weight
            base_appearance_weight: Default appearance weight
            adaptation_rate: How much to adapt (0-1, higher = more adaptive)
            history_window: Window for tracking modality performance
        """
        self.base_weights = {
            'skeleton': base_skeleton_weight,
            'spatial': base_spatial_weight,
            'appearance': base_appearance_weight
        }
        
        self.adaptation_rate = adaptation_rate
        self.history_window = history_window
        
        # Track modality performance history
        self.modality_success_rates = {
            'skeleton': deque(maxlen=history_window),
            'appearance': deque(maxlen=history_window),
            'spatial': deque(maxlen=history_window)
        }
        
        # Statistics
        self.total_adaptations = 0
        self.weight_history = deque(maxlen=1000)
        
        logger.info("Adaptive Weight Manager initialized")
        logger.info(f"  Base weights: Skel={base_skeleton_weight:.2f}, "
                   f"Spat={base_spatial_weight:.2f}, App={base_appearance_weight:.2f}")
        logger.info(f"  Adaptation rate: {adaptation_rate}")
    
    def calculate_adaptive_weights(self,
                                  keypoints: Optional[np.ndarray] = None,
                                  crop_bbox: Optional[Tuple[float, ...]] = None,
                                  position: Optional[Tuple[float, float]] = None,
                                  frame_brightness: Optional[float] = None,
                                  distance_to_camera: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate adaptive weights based on current conditions.
        
        Args:
            keypoints: YOLO keypoints (17, 3) [x, y, confidence]
            crop_bbox: Person crop bounding box [x1, y1, x2, y2]
            position: Person position (x, y)
            frame_brightness: Average frame brightness (0-255)
            distance_to_camera: Distance from camera (pixels)
        
        Returns:
            Dictionary of normalized weights
        """
        # Calculate quality scores
        quality = self._assess_modality_quality(
            keypoints, crop_bbox, position, frame_brightness, distance_to_camera
        )
        
        # Get historical performance
        historical_reliability = self._get_historical_reliability()
        
        # Calculate adjusted weights
        adjusted_weights = {}
        
        for modality in ['skeleton', 'appearance', 'spatial']:
            # Base weight
            base = self.base_weights[modality]
            
            # Quality adjustment
            quality_score = quality.to_dict()[modality]
            quality_factor = 1.0 + self.adaptation_rate * (quality_score - 0.5) * 2
            
            # Historical reliability adjustment
            reliability = historical_reliability.get(modality, 0.5)
            reliability_factor = 0.8 + 0.4 * reliability  # Range [0.8, 1.2]
            
            # Combined adjustment
            adjusted = base * quality_factor * reliability_factor
            adjusted_weights[modality] = max(0.05, adjusted)  # Minimum 5%
        
        # Normalize to sum to 1.0
        total = sum(adjusted_weights.values())
        normalized_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        # Store for statistics
        self.weight_history.append(normalized_weights.copy())
        self.total_adaptations += 1
        
        # Log significant adaptations
        if self._is_significant_adaptation(normalized_weights):
            logger.debug(f"Adaptive weights: {self._format_weights(normalized_weights)}")
            logger.debug(f"  Quality: {quality.to_dict()}")
        
        return normalized_weights
    
    def _assess_modality_quality(self,
                                keypoints: Optional[np.ndarray],
                                crop_bbox: Optional[Tuple[float, ...]],
                                position: Optional[Tuple[float, float]],
                                frame_brightness: Optional[float],
                                distance: Optional[float]) -> ModalityQuality:
        """
        Assess quality of each modality.
        
        Returns:
            ModalityQuality with scores 0-1 (higher = better)
        """
        quality = ModalityQuality()
        
        # 1. SKELETON QUALITY
        if keypoints is not None and keypoints.shape[0] == 17:
            # Keypoint visibility ratio
            confidences = keypoints[:, 2]
            visible_count = np.sum(confidences > 0.5)
            visibility_ratio = visible_count / 17.0
            
            # Average confidence of visible keypoints
            visible_confidences = confidences[confidences > 0.5]
            avg_confidence = np.mean(visible_confidences) if len(visible_confidences) > 0 else 0.0
            
            # Combined skeleton quality
            quality.skeleton_quality = 0.6 * visibility_ratio + 0.4 * avg_confidence
            
            # Penalize if too few keypoints
            if visible_count < 8:
                quality.skeleton_quality *= 0.5
        else:
            quality.skeleton_quality = 0.0
        
        # 2. APPEARANCE QUALITY
        if crop_bbox is not None:
            x1, y1, x2, y2 = crop_bbox
            crop_width = x2 - x1
            crop_height = y2 - y1
            crop_area = crop_width * crop_height
            
            # Size quality (prefer larger crops)
            size_quality = min(1.0, crop_area / (150 * 300))  # Normalize to 150x300
            
            # Aspect ratio quality (prefer person-like aspect ratios)
            aspect_ratio = crop_height / (crop_width + 1e-6)
            ideal_aspect = 2.0  # Person aspect ratio
            aspect_quality = 1.0 - min(1.0, abs(aspect_ratio - ideal_aspect) / ideal_aspect)
            
            # Lighting quality
            lighting_quality = 1.0
            if frame_brightness is not None:
                # Penalize very dark or very bright
                if frame_brightness < 50:
                    lighting_quality = frame_brightness / 50.0
                elif frame_brightness > 200:
                    lighting_quality = (255 - frame_brightness) / 55.0
            
            # Distance penalty
            distance_quality = 1.0
            if distance is not None:
                # Penalize distant people (appearance degrades)
                distance_quality = max(0.3, 1.0 - distance / 500.0)
            
            # Combined appearance quality
            quality.appearance_quality = (
                0.4 * size_quality +
                0.2 * aspect_quality +
                0.2 * lighting_quality +
                0.2 * distance_quality
            )
        else:
            quality.appearance_quality = 0.0
        
        # 3. SPATIAL QUALITY
        # Spatial is always available if position provided
        if position is not None:
            # Spatial quality is generally stable
            quality.spatial_quality = 0.8
        else:
            quality.spatial_quality = 0.0
        
        return quality
    
    def _get_historical_reliability(self) -> Dict[str, float]:
        """
        Get historical reliability scores for each modality.
        
        Returns:
            Dict of reliability scores 0-1
        """
        reliability = {}
        
        for modality in ['skeleton', 'appearance', 'spatial']:
            history = self.modality_success_rates[modality]
            
            if len(history) > 0:
                # Average success rate
                reliability[modality] = np.mean(history)
            else:
                # No history, assume neutral
                reliability[modality] = 0.5
        
        return reliability
    
    def update_modality_success(self, 
                               modality: str,
                               was_successful: bool):
        """
        Update historical success rate for a modality.
        
        Args:
            modality: 'skeleton', 'appearance', or 'spatial'
            was_successful: Whether matching with this modality was successful
        """
        if modality in self.modality_success_rates:
            self.modality_success_rates[modality].append(1.0 if was_successful else 0.0)
    
    def _is_significant_adaptation(self, weights: Dict[str, float]) -> bool:
        """Check if adaptation is significant enough to log."""
        for modality, weight in weights.items():
            base = self.base_weights[modality]
            if abs(weight - base) > 0.10:  # >10% change
                return True
        return False
    
    def _format_weights(self, weights: Dict[str, float]) -> str:
        """Format weights for logging."""
        return f"Skel={weights['skeleton']:.2f}, Spat={weights['spatial']:.2f}, App={weights['appearance']:.2f}"
    
    def get_statistics(self) -> Dict:
        """Get adaptation statistics."""
        if len(self.weight_history) == 0:
            return {
                'total_adaptations': 0,
                'average_weights': self.base_weights.copy()
            }
        
        # Calculate average weights
        avg_weights = {
            'skeleton': np.mean([w['skeleton'] for w in self.weight_history]),
            'spatial': np.mean([w['spatial'] for w in self.weight_history]),
            'appearance': np.mean([w['appearance'] for w in self.weight_history])
        }
        
        # Calculate variance (how much weights change)
        variance = {
            'skeleton': np.var([w['skeleton'] for w in self.weight_history]),
            'spatial': np.var([w['spatial'] for w in self.weight_history]),
            'appearance': np.var([w['appearance'] for w in self.weight_history])
        }
        
        return {
            'total_adaptations': self.total_adaptations,
            'average_weights': avg_weights,
            'weight_variance': variance,
            'base_weights': self.base_weights.copy(),
            'modality_success_rates': {
                mod: np.mean(hist) if len(hist) > 0 else 0.5
                for mod, hist in self.modality_success_rates.items()
            }
        }
