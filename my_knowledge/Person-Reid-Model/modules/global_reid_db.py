"""
global_reid_db.py - Enhanced Global Person Re-Identification Database
======================================================================

Thread-safe global database for multi-camera person re-identification.
NOW WITH HIERARCHICAL MATCHING AND ADAPTIVE THRESHOLDS!

Key Features:
- Thread-safe operations (supports 3-4 simultaneous camera streams)
- ✅ NEW: Hierarchical matching strategy (3 passes)
- ✅ NEW: Adaptive threshold adjustment (context-aware)
- ✅ NEW: Multi-zone person tracking
- ✅ NEW: Temporal decay for spatial penalty
- ✅ NEW: Physical plausibility checks
- Global person ID assignment
- Multi-modal feature storage (skeleton, appearance, spatial)
- Cross-camera matching

Version: 2.0 (Enhanced)
Author: AI Team
Date: 2026-01-05
Previous fixes maintained: Fix #1, #2, #3
"""

import logging
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque

from . import config

logger = logging.getLogger(__name__)


@dataclass
class ZoneTransition:
    """Record of zone transition."""
    from_zone: Optional[int]
    to_zone: int
    timestamp: float
    distance: float


@dataclass
class PersonRecord:
    """
    Enhanced person record across all cameras.
    
    NEW: Multi-zone tracking, movement statistics, quality tracking
    """
    person_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # Multi-modal features (updated across cameras)
    skeleton_features: Optional[np.ndarray] = None
    appearance_features: Optional[np.ndarray] = None
    
    # Camera-specific tracking
    camera_observations: Dict[str, List[Tuple[float, Tuple[float, float]]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    # ✅ NEW: Enhanced spatial tracking
    zones_visited: Set[int] = field(default_factory=set)
    zone_transition_history: deque = field(default_factory=lambda: deque(maxlen=50))
    primary_zones: List[int] = field(default_factory=list)
    is_multi_zone_person: bool = False
    
    # ✅ NEW: Movement tracking
    recent_positions: deque = field(default_factory=lambda: deque(maxlen=20))
    average_speed: float = 0.0  # pixels per second
    
    # Statistics
    total_observations: int = 0
    cameras_seen: set = field(default_factory=set)
    
    # ✅ NEW: Quality tracking
    skeleton_quality_history: deque = field(default_factory=lambda: deque(maxlen=30))
    appearance_quality_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def update_features(self,
                       skeleton_features: Optional[np.ndarray] = None,
                       appearance_features: Optional[np.ndarray] = None,
                       alpha: float = None):
        """Update person features with exponential moving average."""
        if alpha is None:
            alpha = config.GLOBAL_DB.FEATURE_UPDATE_ALPHA
        
        # Update skeleton features
        if skeleton_features is not None:
            if self.skeleton_features is None:
                self.skeleton_features = skeleton_features.copy()
            else:
                self.skeleton_features = (
                    alpha * skeleton_features +
                    (1 - alpha) * self.skeleton_features
                )
                # Re-normalize
                norm = np.linalg.norm(self.skeleton_features)
                if norm > 0:
                    self.skeleton_features = self.skeleton_features / norm
        
        # Update appearance features
        if appearance_features is not None:
            if self.appearance_features is None:
                self.appearance_features = appearance_features.copy()
            else:
                self.appearance_features = (
                    alpha * appearance_features +
                    (1 - alpha) * self.appearance_features
                )
                # Re-normalize
                norm = np.linalg.norm(self.appearance_features)
                if norm > 0:
                    self.appearance_features = self.appearance_features / norm
    
    def add_observation(self,
                       camera_id: str,
                       position: Tuple[float, float],
                       timestamp: Optional[float] = None):
        """Add observation from a camera."""
        if timestamp is None:
            timestamp = time.time()
        
        self.camera_observations[camera_id].append((timestamp, position))
        self.cameras_seen.add(camera_id)
        self.total_observations += 1
        self.last_seen = timestamp
    
    def add_zone_transition(self, from_zone: Optional[int], to_zone: int, position: Tuple[float, float]):
        """✅ NEW: Record zone transition."""
        self.zones_visited.add(to_zone)
        
        # Calculate distance if from_zone known
        distance = 0.0
        if from_zone is not None and len(self.recent_positions) > 0:
            last_pos = self.recent_positions[-1][0]
            distance = np.linalg.norm(np.array(position) - np.array(last_pos))
        
        transition = ZoneTransition(
            from_zone=from_zone,
            to_zone=to_zone,
            timestamp=time.time(),
            distance=distance
        )
        self.zone_transition_history.append(transition)
        
        # Update multi-zone status
        if len(self.zones_visited) >= 3:
            self.is_multi_zone_person = True
    
    def update_movement_stats(self, position: Tuple[float, float], timestamp: float):
        """✅ NEW: Update movement statistics."""
        if len(self.recent_positions) > 0:
            last_pos, last_time = self.recent_positions[-1]
            
            # Calculate speed
            distance = np.linalg.norm(np.array(position) - np.array(last_pos))
            time_diff = timestamp - last_time
            
            if time_diff > 0:
                speed = distance / time_diff
                # Exponential moving average
                self.average_speed = 0.3 * speed + 0.7 * self.average_speed
        
        self.recent_positions.append((position, timestamp))
    
    def get_average_skeleton_quality(self) -> float:
        """✅ NEW: Get average skeleton feature quality."""
        if len(self.skeleton_quality_history) == 0:
            return 0.5
        return np.mean(self.skeleton_quality_history)
    
    def get_average_appearance_quality(self) -> float:
        """✅ NEW: Get average appearance feature quality."""
        if len(self.appearance_quality_history) == 0:
            return 0.5
        return np.mean(self.appearance_quality_history)
    
    def get_recent_position(self, camera_id: str) -> Optional[Tuple[float, float]]:
        """Get most recent position in a specific camera."""
        if camera_id not in self.camera_observations:
            return None
        
        observations = self.camera_observations[camera_id]
        if not observations:
            return None
        
        return observations[-1][1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'person_id': self.person_id,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'total_observations': self.total_observations,
            'cameras_seen': list(self.cameras_seen),
            'zones_visited': list(self.zones_visited),
            'is_multi_zone_person': self.is_multi_zone_person,
            'camera_observation_counts': {
                cam: len(obs) for cam, obs in self.camera_observations.items()
            }
        }


class GlobalReIDDatabase:
    """
    Enhanced global ReID database with hierarchical matching.
    
    ✅ NEW FEATURES:
    1. Hierarchical matching strategy (3 passes)
    2. Context-aware adaptive thresholds
    3. Multi-zone person tracking
    4. Temporal decay for spatial penalty
    5. Physical plausibility checks
    """
    
    def __init__(self,
                 retention_hours: float = None,
                 feature_update_alpha: float = None,
                 cleanup_interval: int = None):
        """Initialize enhanced global ReID database."""
        # Use config defaults if not specified
        if retention_hours is None:
            retention_hours = config.GLOBAL_DB.RETENTION_HOURS
        if feature_update_alpha is None:
            feature_update_alpha = config.GLOBAL_DB.FEATURE_UPDATE_ALPHA
        if cleanup_interval is None:
            cleanup_interval = config.GLOBAL_DB.CLEANUP_INTERVAL
        
        self.retention_seconds = retention_hours * 3600
        self.feature_update_alpha = feature_update_alpha
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._person_records: Dict[int, PersonRecord] = {}
        self._next_person_id = 1
        
        # ✅ NEW: Multi-zone tracking
        self._multi_zone_persons: Set[int] = set()
        
        # Statistics
        self._total_persons = 0
        self._total_observations = 0
        self._frame_count = 0
        self._last_cleanup = 0
        
        # ✅ NEW: Matching statistics
        self._zone_change_matches = 0
        self._multi_zone_matches = 0
        self._hierarchical_matches = 0
        self._threshold_adaptations = 0
        self._skeleton_only_matches = 0
        self._bimodal_matches = 0
        self._full_multimodal_matches = 0
        
        logger.info("Enhanced Global ReID Database initialized (v2.0)")
        logger.info(f"  Retention: {retention_hours} hours")
        logger.info(f"  Feature update alpha: {feature_update_alpha}")
        logger.info("  ✅ NEW: Hierarchical matching enabled")
        logger.info("  ✅ NEW: Adaptive thresholds enabled")
        logger.info("  ✅ NEW: Multi-zone person tracking enabled")
    
    def find_match(self,
                  camera_id: str,
                  skeleton_features: Optional[np.ndarray] = None,
                  appearance_features: Optional[np.ndarray] = None,
                  position: Optional[Tuple[float, float]] = None,
                  current_zone: Optional[int] = None,
                  spatial_tracker=None) -> Optional[Tuple[int, float, Dict[str, Any]]]:
        """
        ✅ ENHANCED: Find matching person with hierarchical strategy.
        
        THREE-PASS MATCHING:
        1. Skeleton-only (threshold=0.85) - Bypasses spatial penalty
        2. Skeleton + Appearance (threshold=0.75) - Standard matching
        3. Full multimodal with adaptive threshold (threshold=0.60-0.80) - Context-aware
        
        Args:
            camera_id: Camera identifier
            skeleton_features: Skeleton features
            appearance_features: Appearance features
            position: (x, y) position in camera frame
            current_zone: Current zone ID (from spatial tracker)
            spatial_tracker: Spatial tracker instance for similarity calculation
        
        Returns:
            Tuple of (person_id, similarity, match_info) or None
            match_info contains: threshold_used, strategy, zone_change, etc.
        """
        with self._lock:
            if len(self._person_records) == 0:
                return None
            
            # ✅ HIERARCHICAL MATCHING STRATEGY
            match = self._hierarchical_match(
                camera_id, skeleton_features, appearance_features,
                position, current_zone, spatial_tracker
            )
            
            if match:
                person_id, similarity, match_info = match
                
                # Update person record
                self._update_person_enhanced(
                    person_id, camera_id, skeleton_features,
                    appearance_features, position, current_zone
                )
                
                return person_id, similarity, match_info
            
            return None
    
    def _hierarchical_match(self,
                           camera_id: str,
                           skeleton_features: Optional[np.ndarray],
                           appearance_features: Optional[np.ndarray],
                           position: Optional[Tuple[float, float]],
                           current_zone: Optional[int],
                           spatial_tracker) -> Optional[Tuple[int, float, Dict[str, Any]]]:
        """
        ✅ NEW: Hierarchical matching strategy with three passes.
        
        PASS 1: Skeleton-only (strict threshold=0.85)
        - Bypasses spatial penalty completely
        - Best for same person in different zones
        - Fastest matching
        
        PASS 2: Skeleton + Appearance (medium threshold=0.75)
        - No spatial component
        - Good for zone changes with good appearance
        
        PASS 3: Full multimodal with adaptive threshold (relaxed threshold=0.60-0.80)
        - Uses all modalities
        - Context-aware threshold adjustment
        - Handles edge cases
        """
        
        # ═══════════════════════════════════════════════════════════════
        # PASS 1: Skeleton-only matching (HIGH threshold)
        # ═══════════════════════════════════════════════════════════════
        if skeleton_features is not None:
            skeleton_match = self._skeleton_only_match(
                skeleton_features,
                threshold=0.85  # Very strict - only strong matches
            )
            
            if skeleton_match:
                person_id, skeleton_sim = skeleton_match
                
                # Verify physical plausibility
                if self._is_plausible_match(person_id, position, camera_id):
                    self._skeleton_only_matches += 1
                    self._hierarchical_matches += 1
                    
                    match_info = {
                        'strategy': 'skeleton_only',
                        'threshold_used': 0.85,
                        'skeleton_similarity': skeleton_sim,
                        'zone_change': self._is_zone_change(person_id, current_zone),
                        'pass': 1
                    }
                    
                    logger.debug(f"✅ PASS 1 (Skeleton-only): Person {person_id}, sim={skeleton_sim:.3f}")
                    return person_id, skeleton_sim, match_info
        
        # ═══════════════════════════════════════════════════════════════
        # PASS 2: Skeleton + Appearance (MEDIUM threshold)
        # ═══════════════════════════════════════════════════════════════
        if skeleton_features is not None and appearance_features is not None:
            bimodal_match = self._bimodal_match(
                skeleton_features,
                appearance_features,
                threshold=0.75  # Standard threshold
            )
            
            if bimodal_match:
                person_id, bimodal_sim, skel_sim, app_sim = bimodal_match
                
                if self._is_plausible_match(person_id, position, camera_id):
                    self._bimodal_matches += 1
                    self._hierarchical_matches += 1
                    
                    match_info = {
                        'strategy': 'bimodal',
                        'threshold_used': 0.75,
                        'skeleton_similarity': skel_sim,
                        'appearance_similarity': app_sim,
                        'zone_change': self._is_zone_change(person_id, current_zone),
                        'pass': 2
                    }
                    
                    logger.debug(f"✅ PASS 2 (Bimodal): Person {person_id}, sim={bimodal_sim:.3f}")
                    return person_id, bimodal_sim, match_info
        
        # ═══════════════════════════════════════════════════════════════
        # PASS 3: Full multimodal with ADAPTIVE threshold (RELAXED)
        # ═══════════════════════════════════════════════════════════════
        full_match = self._full_multimodal_match(
            camera_id,
            skeleton_features,
            appearance_features,
            position,
            current_zone,
            spatial_tracker
        )
        
        if full_match:
            person_id, similarity, threshold_used, similarities = full_match
            
            self._full_multimodal_matches += 1
            
            match_info = {
                'strategy': 'multimodal_adaptive',
                'threshold_used': threshold_used,
                **similarities,
                'zone_change': self._is_zone_change(person_id, current_zone),
                'threshold_adapted': threshold_used != 0.75,
                'pass': 3
            }
            
            logger.debug(f"✅ PASS 3 (Multimodal): Person {person_id}, "
                        f"sim={similarity:.3f}, threshold={threshold_used:.2f}")
            return person_id, similarity, match_info
        
        # No match found in any pass
        return None
    
    def _skeleton_only_match(self,
                           skeleton_features: np.ndarray,
                           threshold: float) -> Optional[Tuple[int, float]]:
        """
        ✅ NEW: Match based on skeleton features only.
        
        This bypasses spatial penalty completely - perfect for zone changes!
        """
        best_match_id = None
        best_similarity = 0.0
        
        for person_id, record in self._person_records.items():
            if record.skeleton_features is None:
                continue
            
            # ✅ FIX #2: Skip zero embeddings
            if np.linalg.norm(record.skeleton_features) < 1e-6:
                continue
            
            # Calculate skeleton similarity
            skeleton_sim = self._cosine_similarity(
                skeleton_features,
                record.skeleton_features
            )
            
            if skeleton_sim > best_similarity:
                best_similarity = skeleton_sim
                best_match_id = person_id
        
        if best_similarity >= threshold:
            return best_match_id, best_similarity
        
        return None
    
    def _bimodal_match(self,
                      skeleton_features: np.ndarray,
                      appearance_features: np.ndarray,
                      threshold: float) -> Optional[Tuple[int, float, float, float]]:
        """
        ✅ NEW: Match based on skeleton + appearance (no spatial).
        
        Weights: 85% skeleton, 15% appearance (skeleton-dominant)
        """
        best_match_id = None
        best_similarity = 0.0
        best_skel_sim = 0.0
        best_app_sim = 0.0
        
        # Weights for bimodal: heavily favor skeleton
        skel_weight = 0.85
        app_weight = 0.15
        
        for person_id, record in self._person_records.items():
            if record.skeleton_features is None or record.appearance_features is None:
                continue
            
            # ✅ FIX #2: Skip zero embeddings
            if (np.linalg.norm(record.skeleton_features) < 1e-6 or
                np.linalg.norm(record.appearance_features) < 1e-6):
                continue
            
            # Calculate similarities
            skel_sim = self._cosine_similarity(skeleton_features, record.skeleton_features)
            app_sim = self._cosine_similarity(appearance_features, record.appearance_features)
            
            # Weighted combination
            combined = skel_weight * skel_sim + app_weight * app_sim
            
            if combined > best_similarity:
                best_similarity = combined
                best_match_id = person_id
                best_skel_sim = skel_sim
                best_app_sim = app_sim
        
        if best_similarity >= threshold:
            return best_match_id, best_similarity, best_skel_sim, best_app_sim
        
        return None
    
    def _full_multimodal_match(self,
                              camera_id: str,
                              skeleton_features: Optional[np.ndarray],
                              appearance_features: Optional[np.ndarray],
                              position: Optional[Tuple[float, float]],
                              current_zone: Optional[int],
                              spatial_tracker) -> Optional[Tuple[int, float, float, Dict[str, float]]]:
        """
        ✅ ENHANCED: Full multimodal match with ADAPTIVE threshold.
        
        Key improvements:
        - Adaptive threshold based on context
        - Temporal decay for spatial penalty
        - Zone-change aware weight adjustment
        """
        best_match_id = None
        best_similarity = 0.0
        best_threshold = config.FUSION.SIMILARITY_THRESHOLD  # Default 0.75
        best_similarities = {}
        
        for person_id, record in self._person_records.items():
            # Calculate individual similarities
            similarities = {}
            
            # 1. Skeleton similarity
            if skeleton_features is not None and record.skeleton_features is not None:
                if np.linalg.norm(record.skeleton_features) > 1e-6:
                    similarities['skeleton'] = self._cosine_similarity(
                        skeleton_features, record.skeleton_features
                    )
            
            # 2. Appearance similarity
            if appearance_features is not None and record.appearance_features is not None:
                if np.linalg.norm(record.appearance_features) > 1e-6:
                    similarities['appearance'] = self._cosine_similarity(
                        appearance_features, record.appearance_features
                    )
            
            # 3. Spatial similarity (with temporal decay)
            if position is not None and spatial_tracker is not None:
                # Get base spatial similarity
                base_spatial = spatial_tracker.get_spatial_similarity(person_id, position)
                
                # ✅ NEW: Apply temporal decay
                time_since_last = time.time() - record.last_seen
                temporal_factor = self._calculate_temporal_decay(time_since_last)
                
                # Adjusted spatial similarity
                similarities['spatial'] = base_spatial * (0.3 + 0.7 * temporal_factor)
            
            # ✅ NEW: Get adaptive weights based on context
            weights = self._get_adaptive_weights(
                person_id, current_zone, record, similarities
            )
            
            # Calculate weighted fusion
            combined = sum(
                weights.get(mod, 0.0) * sim
                for mod, sim in similarities.items()
            )
            
            # ✅ NEW: Get adaptive threshold for this candidate
            threshold = self._get_adaptive_threshold(
                person_id, current_zone, record, similarities
            )
            
            if combined > best_similarity:
                best_similarity = combined
                best_match_id = person_id
                best_threshold = threshold
                best_similarities = similarities.copy()
        
        if best_similarity >= best_threshold:
            return best_match_id, best_similarity, best_threshold, best_similarities
        
        return None
    
    def _get_adaptive_threshold(self,
                               person_id: int,
                               current_zone: Optional[int],
                               record: PersonRecord,
                               similarities: Dict[str, float]) -> float:
        """
        ✅ NEW: Calculate adaptive threshold based on context.
        
        CRITICAL FIX: Lower threshold when:
        1. Zone change detected (-0.05 to -0.08)
        2. Person is multi-zone worker (-0.08)
        3. Strong skeleton evidence (-0.05)
        4. Long time since last seen (-0.03)
        5. Low historical quality (-0.05)
        """
        base_threshold = config.FUSION.SIMILARITY_THRESHOLD  # 0.75
        
        # Factor 1: Zone change penalty reduction
        zone_change_factor = 0.0
        if current_zone is not None:
            if self._is_zone_change(person_id, current_zone):
                if record.is_multi_zone_person:
                    # Multi-zone person: reduce threshold significantly
                    zone_change_factor = -0.08
                    self._multi_zone_matches += 1
                    logger.debug(f"Multi-zone person {person_id}: threshold -0.08")
                else:
                    # Regular person in new zone: moderate reduction
                    zone_change_factor = -0.05
                    self._zone_change_matches += 1
                    logger.debug(f"Zone change for person {person_id}: threshold -0.05")
        
        # Factor 2: Strong skeleton evidence override
        skeleton_factor = 0.0
        if 'skeleton' in similarities and similarities['skeleton'] > 0.85:
            # Very strong skeleton match: reduce threshold
            skeleton_factor = -0.05
            logger.debug(f"Strong skeleton {similarities['skeleton']:.3f}: threshold -0.05")
        
        # Factor 3: Time decay factor
        time_factor = 0.0
        time_since_last = time.time() - record.last_seen
        if time_since_last > 300:  # 5 minutes
            # Haven't seen person recently: reduce threshold
            time_factor = -0.03
            logger.debug(f"Time gap {time_since_last/60:.1f}min: threshold -0.03")
        
        # Factor 4: Quality adjustment
        quality_factor = 0.0
        avg_skeleton_quality = record.get_average_skeleton_quality()
        if avg_skeleton_quality < 0.5:
            # Low historical quality: reduce threshold
            quality_factor = -0.05
            logger.debug(f"Low skeleton quality {avg_skeleton_quality:.2f}: threshold -0.05")
        
        # Combine all factors
        adjusted_threshold = base_threshold + zone_change_factor + skeleton_factor + time_factor + quality_factor
        
        # Clamp to reasonable range [0.60, 0.80]
        adjusted_threshold = max(0.60, min(0.80, adjusted_threshold))
        
        # Track adaptations
        if adjusted_threshold != base_threshold:
            self._threshold_adaptations += 1
            logger.debug(f"🔧 Threshold: {base_threshold:.2f} → {adjusted_threshold:.2f} "
                        f"(zone={zone_change_factor:.2f}, skel={skeleton_factor:.2f}, "
                        f"time={time_factor:.2f}, qual={quality_factor:.2f})")
        
        return adjusted_threshold
    
    def _get_adaptive_weights(self,
                            person_id: int,
                            current_zone: Optional[int],
                            record: PersonRecord,
                            similarities: Dict[str, float]) -> Dict[str, float]:
        """
        ✅ NEW: Calculate adaptive weights based on context.
        
        CRITICAL FIX: Reduce spatial weight when:
        - Zone change detected
        - Person is multi-zone worker
        """
        base_weights = {
            'skeleton': config.FUSION.SKELETON_WEIGHT,      # 0.65
            'spatial': config.FUSION.SPATIAL_WEIGHT,        # 0.25
            'appearance': config.FUSION.APPEARANCE_WEIGHT   # 0.10
        }
        
        # Adjust for zone changes
        if current_zone is not None and self._is_zone_change(person_id, current_zone):
            if record.is_multi_zone_person:
                # Multi-zone person: drastically reduce spatial weight
                base_weights['skeleton'] = 0.75
                base_weights['spatial'] = 0.10
                base_weights['appearance'] = 0.15
                logger.debug(f"Multi-zone weights: Skel=0.75, Spat=0.10, App=0.15")
            else:
                # Regular person in new zone: moderately reduce spatial weight
                base_weights['skeleton'] = 0.70
                base_weights['spatial'] = 0.15
                base_weights['appearance'] = 0.15
                logger.debug(f"Zone change weights: Skel=0.70, Spat=0.15, App=0.15")
        
        # Normalize available weights
        available_weights = {
            mod: weight
            for mod, weight in base_weights.items()
            if mod in similarities
        }
        
        total = sum(available_weights.values())
        if total > 0:
            normalized = {k: v/total for k, v in available_weights.items()}
            return normalized
        
        return base_weights
    
    def _calculate_temporal_decay(self, time_since_last_seen: float) -> float:
        """
        ✅ NEW: Calculate temporal decay factor (0-1).
        
        Spatial expectations decay over time:
        - 0 seconds: 1.0 (full spatial weight)
        - 5 minutes: 0.6
        - 15 minutes: 0.3
        - 30+ minutes: 0.1
        """
        tau = 300.0  # 5 minutes
        decay = np.exp(-time_since_last_seen / tau)
        return max(0.1, decay)  # Minimum 0.1
    
    def _is_zone_change(self, person_id: int, current_zone: Optional[int]) -> bool:
        """✅ NEW: Check if person is in a different zone than usual."""
        if current_zone is None:
            return False
        
        record = self._person_records.get(person_id)
        if record is None:
            return False
        
        # Check recent zone history
        if len(record.zone_transition_history) > 0:
            recent_zones = [t.to_zone for t in list(record.zone_transition_history)[-5:]]
            
            # If current zone not in recent zones, it's a change
            if current_zone not in recent_zones:
                return True
        
        return False
    
    def _is_plausible_match(self,
                          person_id: int,
                          position: Optional[Tuple[float, float]],
                          camera_id: str) -> bool:
        """
        ✅ NEW: Check if match is physically plausible.
        
        Prevents matching if person would need to teleport.
        Max speed: 10 pixels/second (generous for human movement)
        """
        if position is None:
            return True
        
        record = self._person_records.get(person_id)
        if record is None or len(record.recent_positions) == 0:
            return True
        
        # Get last known position and time
        last_pos, last_time = record.recent_positions[-1]
        current_time = time.time()
        time_diff = current_time - last_time
        
        # Calculate distance
        distance = np.linalg.norm(np.array(position) - np.array(last_pos))
        
        # Check if physically possible
        max_speed = 10.0  # pixels per second (generous)
        required_speed = distance / (time_diff + 1e-6)
        
        if required_speed > max_speed and time_diff < 60:  # Within 1 minute
            logger.warning(f"⚠️ Implausible match rejected: Person {person_id} "
                         f"would need {required_speed:.1f} px/s (max: {max_speed:.1f})")
            return False
        
        return True
    
    def _update_person_enhanced(self,
                              person_id: int,
                              camera_id: str,
                              skeleton_features: Optional[np.ndarray],
                              appearance_features: Optional[np.ndarray],
                              position: Optional[Tuple[float, float]],
                              current_zone: Optional[int]):
        """✅ ENHANCED: Update person record with zone tracking."""
        record = self._person_records[person_id]
        
        # Update features
        record.update_features(skeleton_features, appearance_features)
        
        # Update position and movement
        if position is not None:
            record.update_movement_stats(position, time.time())
            record.add_observation(camera_id, position)
        
        # ✅ NEW: Update zone tracking
        if current_zone is not None:
            # Get previous zone
            prev_zone = None
            if len(record.zone_transition_history) > 0:
                prev_zone = record.zone_transition_history[-1].to_zone
            
            # Record transition if zone changed
            if prev_zone != current_zone:
                record.add_zone_transition(prev_zone, current_zone, position or (0, 0))
                
                # Update multi-zone tracking
                if record.is_multi_zone_person:
                    self._multi_zone_persons.add(person_id)
        
        # Update timestamps
        record.last_seen = time.time()
    
    def add_person(self,
                  camera_id: str,
                  skeleton_features: Optional[np.ndarray] = None,
                  appearance_features: Optional[np.ndarray] = None,
                  position: Optional[Tuple[float, float]] = None,
                  current_zone: Optional[int] = None) -> int:
        """✅ ENHANCED: Add new person with zone tracking."""
        with self._lock:
            person_id = self._next_person_id
            self._next_person_id += 1
            
            # Create enhanced record
            record = PersonRecord(person_id=person_id)
            
            # Initialize features
            if skeleton_features is not None:
                record.skeleton_features = skeleton_features.copy()
            if appearance_features is not None:
                record.appearance_features = appearance_features.copy()
            
            # Initialize position tracking
            if position is not None:
                record.update_movement_stats(position, time.time())
                record.add_observation(camera_id, position)
            
            # ✅ NEW: Initialize zone tracking
            if current_zone is not None:
                record.add_zone_transition(None, current_zone, position or (0, 0))
            
            # Store record
            self._person_records[person_id] = record
            self._total_persons += 1
            
            logger.info(f"✨ New person: ID={person_id}, camera={camera_id}, zone={current_zone}")
            
            return person_id
    
    def update_person(self,
                     person_id: int,
                     camera_id: str,
                     skeleton_features: Optional[np.ndarray] = None,
                     appearance_features: Optional[np.ndarray] = None,
                     position: Optional[Tuple[float, float]] = None,
                     current_zone: Optional[int] = None):
        """Update existing person record."""
        with self._lock:
            if person_id not in self._person_records:
                logger.warning(f"Person {person_id} not found in database")
                return
            
            self._update_person_enhanced(
                person_id, camera_id, skeleton_features,
                appearance_features, position, current_zone
            )
    
    def _cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        return float((cosine_sim + 1.0) / 2.0)
    
    def get_person(self, person_id: int) -> Optional[PersonRecord]:
        """Get person record by ID."""
        with self._lock:
            return self._person_records.get(person_id)
    
    def get_all_persons(self) -> Dict[int, PersonRecord]:
        """Get all person records."""
        with self._lock:
            return self._person_records.copy()
    
    def cleanup_old_records(self):
        """Remove old person records beyond retention window."""
        self._frame_count += 1
        
        # Check cleanup interval
        if (self._frame_count - self._last_cleanup) < self.cleanup_interval:
            return
        
        with self._lock:
            current_time = time.time()
            old_ids = []
            
            for person_id, record in self._person_records.items():
                if current_time - record.last_seen > self.retention_seconds:
                    old_ids.append(person_id)
            
            # Remove old records
            for person_id in old_ids:
                del self._person_records[person_id]
                self._multi_zone_persons.discard(person_id)
            
            if old_ids:
                logger.info(f"🧹 Cleaned up {len(old_ids)} old records")
            
            self._last_cleanup = self._frame_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """✅ ENHANCED: Get database statistics with new metrics."""
        with self._lock:
            stats = {
                'total_persons_tracked': self._total_persons,
                'active_persons': len(self._person_records),
                'total_observations': self._total_observations,
                'retention_hours': self.retention_seconds / 3600,
                'cameras_active': len(set(
                    cam for record in self._person_records.values()
                    for cam in record.cameras_seen
                )),
                # ✅ NEW: Multi-zone statistics
                'multi_zone_persons': len(self._multi_zone_persons),
                # ✅ NEW: Matching strategy statistics
                'matching_stats': {
                    'skeleton_only_matches': self._skeleton_only_matches,
                    'bimodal_matches': self._bimodal_matches,
                    'full_multimodal_matches': self._full_multimodal_matches,
                    'zone_change_matches': self._zone_change_matches,
                    'multi_zone_matches': self._multi_zone_matches,
                    'threshold_adaptations': self._threshold_adaptations,
                    'total_hierarchical': self._hierarchical_matches
                }
            }
            
            # Per-camera breakdown
            camera_counts = defaultdict(int)
            for record in self._person_records.values():
                for cam in record.cameras_seen:
                    camera_counts[cam] += 1
            
            stats['persons_per_camera'] = dict(camera_counts)
            
            return stats
    
    def reset(self):
        """Reset database (for testing)."""
        with self._lock:
            self._person_records.clear()
            self._multi_zone_persons.clear()
            self._next_person_id = 1
            self._total_persons = 0
            self._total_observations = 0
            self._frame_count = 0
            self._last_cleanup = 0
            self._zone_change_matches = 0
            self._multi_zone_matches = 0
            self._hierarchical_matches = 0
            self._threshold_adaptations = 0
            logger.info("🔄 Database reset")
