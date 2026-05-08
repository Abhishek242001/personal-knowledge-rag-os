"""
temporal_smoother.py - Temporal Consistency via Smoothing
=========================================================

Smooth similarity scores and feature vectors over time.
Estimated accuracy improvement: +2-3%

Author: AI Team
Date: 2026-01-05
Version: 2.0
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Apply temporal smoothing to reduce frame-to-frame jitter.
    
    Smooths:
    1. Similarity scores (reduce false matches)
    2. Feature vectors (stabilize representations)
    3. Position predictions (reduce tracking noise)
    """
    
    def __init__(self,
                 similarity_window: int = 5,
                 feature_window: int = 3,
                 position_window: int = 10,
                 alpha: float = 0.7):
        """
        Initialize temporal smoother.
        
        Args:
            similarity_window: Window size for similarity smoothing
            feature_window: Window size for feature smoothing
            position_window: Window size for position smoothing
            alpha: EMA weight for new values (0-1, higher = less smoothing)
        """
        self.similarity_window = similarity_window
        self.feature_window = feature_window
        self.position_window = position_window
        self.alpha = alpha
        
        # History buffers per person
        self.similarity_history: Dict[Tuple[int, int], deque] = {}  # (person1, person2) -> scores
        self.feature_history: Dict[int, deque] = {}  # person_id -> features
        self.position_history: Dict[int, deque] = {}  # person_id -> positions
        
        logger.info("Temporal Smoother initialized")
        logger.info(f"  Windows: sim={similarity_window}, feat={feature_window}, pos={position_window}")
        logger.info(f"  EMA alpha: {alpha}")
    
    def smooth_similarity(self,
                         person1_id: int,
                         person2_id: int,
                         current_similarity: float) -> float:
        """
        Smooth similarity score between two people over time.
        
        Args:
            person1_id: First person ID
            person2_id: Second person ID
            current_similarity: Current frame similarity
        
        Returns:
            Smoothed similarity score
        """
        key = tuple(sorted([person1_id, person2_id]))
        
        # Initialize history if needed
        if key not in self.similarity_history:
            self.similarity_history[key] = deque(maxlen=self.similarity_window)
        
        history = self.similarity_history[key]
        history.append(current_similarity)
        
        if len(history) == 1:
            return current_similarity
        
        # Exponential moving average
        smoothed = self.alpha * current_similarity + (1 - self.alpha) * np.mean(list(history)[:-1])
        
        return float(smoothed)
    
    def smooth_features(self,
                       person_id: int,
                       current_features: np.ndarray) -> np.ndarray:
        """
        Smooth feature vector over time.
        
        Args:
            person_id: Person ID
            current_features: Current frame features
        
        Returns:
            Smoothed feature vector
        """
        # Initialize history if needed
        if person_id not in self.feature_history:
            self.feature_history[person_id] = deque(maxlen=self.feature_window)
        
        history = self.feature_history[person_id]
        history.append(current_features.copy())
        
        if len(history) == 1:
            return current_features
        
        # Weighted average of recent features
        weights = np.linspace(0.5, 1.0, len(history))  # More weight to recent
        weights = weights / weights.sum()
        
        smoothed = np.zeros_like(current_features)
        for w, feat in zip(weights, history):
            smoothed += w * feat
        
        # Renormalize (important for cosine similarity)
        norm = np.linalg.norm(smoothed)
        if norm > 0:
            smoothed = smoothed / norm
        
        return smoothed
    
    def smooth_position(self,
                       person_id: int,
                       current_position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Smooth position over time (reduces jitter).
        
        Args:
            person_id: Person ID
            current_position: Current (x, y) position
        
        Returns:
            Smoothed position
        """
        # Initialize history if needed
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=self.position_window)
        
        history = self.position_history[person_id]
        history.append(current_position)
        
        if len(history) == 1:
            return current_position
        
        # Simple moving average
        positions_array = np.array(list(history))
        smoothed = np.mean(positions_array, axis=0)
        
        return tuple(smoothed)
    
    def predict_next_position(self,
                             person_id: int) -> Optional[Tuple[float, float]]:
        """
        Predict next position based on velocity.
        
        Args:
            person_id: Person ID
        
        Returns:
            Predicted (x, y) position or None
        """
        if person_id not in self.position_history:
            return None
        
        history = self.position_history[person_id]
        
        if len(history) < 3:
            return None
        
        # Simple linear extrapolation
        positions = np.array(list(history))
        
        # Calculate velocity (last 3 positions)
        recent = positions[-3:]
        velocity = recent[-1] - recent[0]
        velocity = velocity / 2.0  # Average over 2 frames
        
        # Predict next position
        predicted = positions[-1] + velocity
        
        return tuple(predicted)
    
    def clear_person_history(self, person_id: int):
        """Clear history for a person who disappeared."""
        if person_id in self.feature_history:
            del self.feature_history[person_id]
        if person_id in self.position_history:
            del self.position_history[person_id]
        
        # Clear similarity history involving this person
        keys_to_remove = [
            key for key in self.similarity_history.keys()
            if person_id in key
        ]
        for key in keys_to_remove:
            del self.similarity_history[key]
    
    def get_statistics(self) -> Dict:
        """Get smoother statistics."""
        return {
            'tracked_persons': len(self.feature_history),
            'similarity_pairs': len(self.similarity_history),
            'position_tracked': len(self.position_history),
            'windows': {
                'similarity': self.similarity_window,
                'feature': self.feature_window,
                'position': self.position_window
            },
            'alpha': self.alpha
        }
