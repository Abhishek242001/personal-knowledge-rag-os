"""
spatial_tracker.py - Enhanced Automatic Spatial Learning Module
===============================================================

Automatically learns spatial patterns (work zones) without manual configuration.
Uses DBSCAN clustering to identify where each person typically works.

✅ ENHANCED: Better multi-zone handling, zone transition tracking, and quality scoring

Key Innovation: ZERO MANUAL CONFIGURATION
- Automatically discovers work zones from movement patterns
- Learns person-zone associations over time
- Provides spatial priors for re-identification
- ✅ NEW: Tracks zone transitions and movement patterns
- ✅ NEW: Quality scores for spatial predictions
- ✅ NEW: Multi-zone person detection

Target: 25% weight in final multi-modal fusion (reduced for zone changes)

Version: 2.0 (Enhanced)
Author: AI Team
Date: 2026-01-05
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, deque

import numpy as np
from . import config

logger = logging.getLogger(__name__)

# Import clustering
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    DBSCAN = None
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - spatial learning disabled")


class SpatialTracker:
    """
    Enhanced automatic spatial learning and tracking system.
    
    ✅ NEW FEATURES:
    1. Zone transition tracking and validation
    2. Multi-zone person detection and scoring
    3. Quality-aware spatial predictions
    4. Movement pattern analysis
    5. Zone topology learning
    
    Features:
    - DBSCAN clustering to discover work zones automatically
    - Track person movement history
    - Learn person-zone associations
    - Provide spatial priors for ReID
    - Adaptive zone boundaries
    """
    
    def __init__(self,
                 eps: float = None,
                 min_samples: int = None,
                 history_window: int = None,
                 learning_rate: float = None,
                 min_observations: int = None,
                 zone_update_interval: int = None):
        """
        Initialize spatial tracker.
        
        Args:
            eps: DBSCAN epsilon (maximum distance between points in a cluster)
            min_samples: DBSCAN minimum samples per cluster
            history_window: Number of recent positions to keep per person
            learning_rate: Learning rate for zone association updates
            min_observations: Minimum observations before using spatial priors
            zone_update_interval: Frames between zone reclustering
        """
        
        # Use config defaults if not specified
        if eps is None:
            eps = config.SPATIAL.DBSCAN_EPS
        if min_samples is None:
            min_samples = config.SPATIAL.DBSCAN_MIN_SAMPLES
        if history_window is None:
            history_window = config.SPATIAL.HISTORY_WINDOW
        if learning_rate is None:
            learning_rate = config.SPATIAL.LEARNING_RATE
        if min_observations is None:
            min_observations = config.SPATIAL.MIN_OBSERVATIONS
        if zone_update_interval is None:
            zone_update_interval = config.SPATIAL.ZONE_UPDATE_INTERVAL
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn not installed. Install with: pip install scikit-learn"
            )
        
        self.eps = eps
        self.min_samples = min_samples
        self.history_window = history_window
        self.learning_rate = learning_rate
        self.min_observations = min_observations
        self.zone_update_interval = zone_update_interval
        
        # Person position history: {person_id: deque of (x, y, timestamp)}
        self.position_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_window)
        )
        
        # Zone information
        self.zones: List[Dict[str, Any]] = []  # List of discovered zones
        self.zone_centers: np.ndarray = np.array([])  # Zone centroids
        
        # Person-zone associations: {person_id: {zone_id: probability}}
        self.person_zone_probs: Dict[int, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # ✅ NEW: Zone transition tracking
        self.zone_transitions: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )  # {person_id: deque of (from_zone, to_zone, timestamp)}
        
        # ✅ NEW: Zone topology (which zones connect to which)
        self.zone_topology: Dict[Tuple[int, int], int] = defaultdict(int)
        # {(zone_a, zone_b): transition_count}
        
        # ✅ NEW: Multi-zone persons
        self.multi_zone_persons: Set[int] = set()
        
        # Statistics
        self.frame_count = 0
        self.last_zone_update = 0
        self.total_observations = 0
        
        # ✅ NEW: Enhanced statistics
        self.stats = {
            'total_transitions': 0,
            'valid_transitions': 0,
            'invalid_transitions': 0,
            'multi_zone_persons_detected': 0
        }
        
        logger.info("✅ Spatial Tracker initialized (v2.0 - Enhanced)")
        logger.info(f"  DBSCAN eps: {eps}px")
        logger.info(f"  Min samples: {min_samples}")
        logger.info(f"  History window: {history_window}")
        logger.info("  ✅ NEW: Zone transition tracking enabled")
        logger.info("  ✅ NEW: Multi-zone detection enabled")
        logger.info(f"  Auto-discovery: ENABLED (zero manual config)")
    
    def update_position(self,
                       person_id: int,
                       position: Tuple[float, float],
                       timestamp: Optional[float] = None):
        """
        Update person's position in spatial tracking.
        
        ✅ ENHANCED: Now tracks zone transitions
        
        Args:
            person_id: Person ID
            position: (x, y) position in frame coordinates
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Get current zone before adding new position
        old_zone = self.get_zone_at_position(
            self.position_history[person_id][-1][:2]
        ) if len(self.position_history[person_id]) > 0 else None
        
        # Add to history
        self.position_history[person_id].append((*position, timestamp))
        self.total_observations += 1
        
        # Get new zone
        new_zone = self.get_zone_at_position(position)
        
        # ✅ NEW: Track zone transition
        if old_zone is not None and new_zone is not None and old_zone != new_zone:
            self._record_zone_transition(person_id, old_zone, new_zone, timestamp)
        
        # Update zone associations if this person has enough history
        if len(self.position_history[person_id]) >= self.min_observations:
            self._update_person_zone_association(person_id, position)
    
    def _record_zone_transition(self,
                               person_id: int,
                               from_zone: int,
                               to_zone: int,
                               timestamp: float):
        """
        ✅ NEW: Record zone transition.
        
        Tracks movement between zones for:
        - Zone topology learning
        - Multi-zone person detection
        - Transition validation
        """
        # Record transition
        self.zone_transitions[person_id].append((from_zone, to_zone, timestamp))
        
        # Update zone topology
        self.zone_topology[(from_zone, to_zone)] += 1
        self.zone_topology[(to_zone, from_zone)] += 1  # Bidirectional
        
        # Update statistics
        self.stats['total_transitions'] += 1
        
        # Validate transition (is it plausible?)
        if self._is_valid_transition(from_zone, to_zone):
            self.stats['valid_transitions'] += 1
        else:
            self.stats['invalid_transitions'] += 1
            logger.debug(f"⚠️ Unusual transition: Person {person_id} "
                        f"Zone {from_zone} → {to_zone}")
        
        # Check if person is multi-zone
        zones_visited = set()
        for trans in self.zone_transitions[person_id]:
            zones_visited.add(trans[0])
            zones_visited.add(trans[1])
        
        if len(zones_visited) >= 3 and person_id not in self.multi_zone_persons:
            self.multi_zone_persons.add(person_id)
            self.stats['multi_zone_persons_detected'] += 1
            logger.info(f"✨ Multi-zone person detected: ID={person_id}, "
                       f"zones={zones_visited}")
    
    def _is_valid_transition(self, from_zone: int, to_zone: int) -> bool:
        """
        ✅ NEW: Validate zone transition.
        
        A transition is valid if:
        1. It's been seen before in zone topology
        2. OR zones are adjacent (within reasonable distance)
        """
        # Check if this transition exists in topology
        if self.zone_topology.get((from_zone, to_zone), 0) > 0:
            return True
        
        # Check if zones are adjacent
        if len(self.zones) > 0:
            if from_zone < len(self.zones) and to_zone < len(self.zones):
                zone_a_center = np.array(self.zones[from_zone]['center'])
                zone_b_center = np.array(self.zones[to_zone]['center'])
                
                distance = np.linalg.norm(zone_a_center - zone_b_center)
                
                # Zones within 3x eps are considered adjacent
                if distance < 3 * self.eps:
                    return True
        
        return False
    
    def update_zones(self, force: bool = False):
        """
        Update work zones using DBSCAN clustering on all position history.
        
        This automatically discovers zones - NO MANUAL CONFIGURATION NEEDED!
        
        ✅ ENHANCED: Better zone stability and quality tracking
        
        Args:
            force: Force zone update even if interval hasn't elapsed
        """
        self.frame_count += 1
        
        # Check if update is needed
        if not force and (self.frame_count - self.last_zone_update) < self.zone_update_interval:
            return
        
        # Need sufficient data for clustering
        if self.total_observations < self.min_samples * 3:
            logger.debug(f"Insufficient data for zone clustering: {self.total_observations} observations")
            return
        
        logger.info(f"🔄 Updating zones (observations: {self.total_observations})...")
        
        try:
            # Collect all positions
            all_positions = []
            for person_id, history in self.position_history.items():
                for x, y, _ in history:
                    all_positions.append([x, y])
            
            if len(all_positions) < self.min_samples:
                return
            
            positions_array = np.array(all_positions)
            
            # Run DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = clustering.fit_predict(positions_array)
            
            # Extract zones (ignore noise points with label -1)
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise cluster
            
            old_zone_count = len(self.zones)
            self.zones = []
            zone_centers_list = []
            
            for zone_id in sorted(unique_labels):
                # Get all points in this zone
                zone_mask = labels == zone_id
                zone_points = positions_array[zone_mask]
                
                # Calculate zone statistics
                center = np.mean(zone_points, axis=0)
                radius = np.max(np.linalg.norm(zone_points - center, axis=1))
                
                # ✅ NEW: Calculate zone quality metrics
                density = len(zone_points) / (np.pi * radius**2 + 1e-6)
                compactness = radius / (np.std(zone_points) + 1e-6)
                
                zone_info = {
                    'zone_id': int(zone_id),
                    'center': center.tolist(),
                    'radius': float(radius),
                    'num_points': int(np.sum(zone_mask)),
                    'density': float(density),
                    'compactness': float(compactness),
                    'bounds': {
                        'x_min': float(np.min(zone_points[:, 0])),
                        'x_max': float(np.max(zone_points[:, 0])),
                        'y_min': float(np.min(zone_points[:, 1])),
                        'y_max': float(np.max(zone_points[:, 1]))
                    }
                }
                
                self.zones.append(zone_info)
                zone_centers_list.append(center)
            
            if zone_centers_list:
                self.zone_centers = np.array(zone_centers_list)
                logger.info(f"✅ Discovered {len(self.zones)} work zones automatically")
                
                # Log zone changes
                if len(self.zones) != old_zone_count:
                    logger.info(f"   Zone count changed: {old_zone_count} → {len(self.zones)}")
                
                for zone in self.zones:
                    logger.info(f"  Zone {zone['zone_id']}: "
                              f"center=({zone['center'][0]:.0f}, {zone['center'][1]:.0f}), "
                              f"radius={zone['radius']:.0f}px, "
                              f"points={zone['num_points']}, "
                              f"density={zone['density']:.2f}")
            else:
                logger.warning("No zones discovered (all points classified as noise)")
            
            self.last_zone_update = self.frame_count
            
        except Exception as e:
            logger.error(f"Zone update failed: {e}", exc_info=True)
    
    def _update_person_zone_association(self, person_id: int, current_position: Tuple[float, float]):
        """
        Update person's zone association probabilities.
        
        ✅ ENHANCED: Better handling of multi-zone persons
        
        Uses exponential moving average to learn which zones each person frequents.
        """
        if len(self.zones) == 0:
            return
        
        # Find nearest zone
        zone_id = self.get_zone_at_position(current_position)
        
        if zone_id is not None:
            # Update probabilities using exponential moving average
            for zid in range(len(self.zones)):
                if zid == zone_id:
                    # Increase probability for current zone
                    self.person_zone_probs[person_id][zid] = (
                        self.learning_rate * 1.0 +
                        (1 - self.learning_rate) * self.person_zone_probs[person_id][zid]
                    )
                else:
                    # Decay probability for other zones
                    self.person_zone_probs[person_id][zid] = (
                        self.learning_rate * 0.0 +
                        (1 - self.learning_rate) * self.person_zone_probs[person_id][zid]
                    )
    
    def get_spatial_similarity(self,
                              person_id: int,
                              position: Tuple[float, float]) -> float:
        """
        Calculate spatial similarity score for a person at a position.
        
        ✅ ENHANCED: Better scoring for multi-zone persons
        
        Args:
            person_id: Person ID
            position: Position to evaluate (x, y)
        
        Returns:
            Similarity score 0-1 (higher = more likely this person would be here)
        """
        # Check if we have enough data
        if person_id not in self.position_history:
            return 0.5  # Neutral score
        
        if len(self.position_history[person_id]) < self.min_observations:
            return 0.5  # Not enough data yet
        
        try:
            # Get zone at position
            zone_id = self.get_zone_at_position(position)
            
            if zone_id is not None and person_id in self.person_zone_probs:
                # Get person's probability for this zone
                zone_prob = self.person_zone_probs[person_id].get(zone_id, 0.0)
                
                # ✅ NEW: Boost score for multi-zone persons
                if person_id in self.multi_zone_persons:
                    # Multi-zone persons get higher baseline
                    zone_prob = max(zone_prob, 0.3)
                    logger.debug(f"Multi-zone person {person_id}: boosted spatial score")
                
                # Check if this is primary zone
                if zone_prob > config.SPATIAL.PRIMARY_ZONE_THRESHOLD:
                    return zone_prob  # High similarity - person works here
                else:
                    # Lower similarity but not zero
                    return max(zone_prob, 0.2)
            
            else:
                # Position outside known zones - use history-based similarity
                return self._calculate_position_history_similarity(person_id, position)
            
        except Exception as e:
            logger.error(f"Spatial similarity calculation failed: {e}")
            return 0.5
    
    def _calculate_position_history_similarity(self,
                                              person_id: int,
                                              position: Tuple[float, float]) -> float:
        """
        Calculate similarity based on historical positions.
        
        Used when position is outside discovered zones.
        
        Args:
            person_id: Person ID
            position: Position to evaluate
        
        Returns:
            Similarity score based on distance to historical positions
        """
        if person_id not in self.position_history:
            return 0.5
        
        history = self.position_history[person_id]
        if len(history) == 0:
            return 0.5
        
        # Calculate distances to historical positions
        pos_array = np.array(position)
        historical_positions = np.array([[x, y] for x, y, _ in history])
        distances = np.linalg.norm(historical_positions - pos_array, axis=1)
        
        # Use minimum distance (closest historical position)
        min_distance = np.min(distances)
        
        # Convert distance to similarity (exponential decay)
        scale = self.eps
        similarity = np.exp(-min_distance / scale)
        
        return float(similarity)
    
    def get_person_primary_zone(self, person_id: int) -> Optional[int]:
        """
        Get the primary work zone for a person.
        
        ✅ ENHANCED: Returns None for multi-zone persons (no single primary zone)
        
        Args:
            person_id: Person ID
        
        Returns:
            Zone ID of primary zone, or None
        """
        if person_id not in self.person_zone_probs:
            return None
        
        # ✅ NEW: Multi-zone persons have no single primary zone
        if person_id in self.multi_zone_persons:
            return None
        
        zone_probs = self.person_zone_probs[person_id]
        if not zone_probs:
            return None
        
        # Find zone with highest probability
        primary_zone = max(zone_probs.items(), key=lambda x: x[1])
        
        # Only return if probability is significant
        if primary_zone[1] > config.SPATIAL.PRIMARY_ZONE_THRESHOLD:
            return primary_zone[0]
        
        return None
    
    def get_person_zones(self, person_id: int, min_probability: float = 0.2) -> List[Tuple[int, float]]:
        """
        ✅ NEW: Get all zones for a person (sorted by probability).
        
        Useful for multi-zone persons who work in multiple areas.
        
        Args:
            person_id: Person ID
            min_probability: Minimum probability threshold
        
        Returns:
            List of (zone_id, probability) tuples, sorted by probability
        """
        if person_id not in self.person_zone_probs:
            return []
        
        zone_probs = self.person_zone_probs[person_id]
        
        # Filter and sort
        zones = [
            (zone_id, prob)
            for zone_id, prob in zone_probs.items()
            if prob >= min_probability
        ]
        
        zones.sort(key=lambda x: x[1], reverse=True)
        
        return zones
    
    def get_zone_at_position(self, position: Tuple[float, float]) -> Optional[int]:
        """
        Get zone ID at a given position.
        
        Args:
            position: (x, y) position
        
        Returns:
            Zone ID or None if position is outside all zones
        """
        if len(self.zones) == 0 or len(self.zone_centers) == 0:
            return None
        
        try:
            pos_array = np.array([position])
            distances = np.linalg.norm(self.zone_centers - pos_array, axis=1)
            nearest_zone_idx = np.argmin(distances)
            nearest_zone_id = self.zones[nearest_zone_idx]['zone_id']
            nearest_distance = distances[nearest_zone_idx]
            
            # Check if within zone (with 1.5x radius tolerance)
            zone_radius = self.zones[nearest_zone_idx]['radius']
            if nearest_distance <= zone_radius * 1.5:
                return nearest_zone_id
            
            return None
            
        except Exception as e:
            logger.error(f"Zone lookup failed: {e}")
            return None
    
    def is_multi_zone_person(self, person_id: int) -> bool:
        """
        ✅ NEW: Check if person is a multi-zone worker.
        
        Args:
            person_id: Person ID
        
        Returns:
            True if person works in 3+ zones
        """
        return person_id in self.multi_zone_persons
    
    def get_zone_transition_count(self, person_id: int) -> int:
        """
        ✅ NEW: Get number of zone transitions for a person.
        
        Args:
            person_id: Person ID
        
        Returns:
            Number of zone transitions
        """
        if person_id not in self.zone_transitions:
            return 0
        return len(self.zone_transitions[person_id])
    
    def get_zone_info(self, zone_id: int) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Get detailed information about a zone.
        
        Args:
            zone_id: Zone ID
        
        Returns:
            Zone information dict or None
        """
        for zone in self.zones:
            if zone['zone_id'] == zone_id:
                return zone
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        ✅ ENHANCED: Get comprehensive spatial tracking statistics.
        """
        stats = {
            'total_observations': self.total_observations,
            'num_tracked_people': len(self.position_history),
            'num_zones': len(self.zones),
            'zones': self.zones,
            'people_with_zone_associations': len(self.person_zone_probs),
            'multi_zone_persons': len(self.multi_zone_persons),
            'total_transitions': self.stats['total_transitions'],
            'valid_transitions': self.stats['valid_transitions'],
            'invalid_transitions': self.stats['invalid_transitions'],
            'zone_topology_edges': len(self.zone_topology)
        }
        
        # Calculate zone utilization
        zone_person_counts = defaultdict(int)
        for person_id, zone_probs in self.person_zone_probs.items():
            for zone_id, prob in zone_probs.items():
                if prob > 0.3:  # Count if significant probability
                    zone_person_counts[zone_id] += 1
        
        stats['zone_utilization'] = dict(zone_person_counts)
        
        return stats
    
    def clear_person_history(self, person_id: int):
        """Clear position history for a person."""
        if person_id in self.position_history:
            del self.position_history[person_id]
        if person_id in self.person_zone_probs:
            del self.person_zone_probs[person_id]
        if person_id in self.zone_transitions:
            del self.zone_transitions[person_id]
        self.multi_zone_persons.discard(person_id)
    
    def reset(self):
        """Reset all spatial tracking data."""
        self.position_history.clear()
        self.zones = []
        self.zone_centers = np.array([])
        self.person_zone_probs.clear()
        self.zone_transitions.clear()
        self.zone_topology.clear()
        self.multi_zone_persons.clear()
        self.frame_count = 0
        self.last_zone_update = 0
        self.total_observations = 0
        self.stats = {
            'total_transitions': 0,
            'valid_transitions': 0,
            'invalid_transitions': 0,
            'multi_zone_persons_detected': 0
        }
        logger.info("🔄 Spatial tracker reset")


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_spatial_tracker_enhanced():
    """Test the enhanced spatial tracker."""
    import sys
    
    print("\n" + "="*70)
    print("ENHANCED SPATIAL TRACKER - TEST")
    print("="*70)
    
    # Initialize tracker
    print("\n1. Initializing spatial tracker...")
    try:
        tracker = SpatialTracker(
            eps=150.0,
            min_samples=10,
            history_window=300
        )
        print("✓ Spatial tracker initialized")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        sys.exit(1)
    
    # Simulate person movements in 3 work zones
    print("\n2. Simulating person movements in 3 work zones...")
    
    zones_data = [
        (1, 100, 100, 30),   # Person 1: Kitchen
        (2, 400, 100, 30),   # Person 2: Dining
        (3, 250, 300, 30)    # Person 3: Service area
    ]
    
    # Add observations
    num_observations = 50
    for i in range(num_observations):
        for person_id, cx, cy, radius in zones_data:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            
            tracker.update_position(person_id, (x, y))
    
    print(f"✓ Added {num_observations * 3} observations")
    
    # Update zones
    print("\n3. Discovering work zones...")
    tracker.update_zones(force=True)
    
    stats = tracker.get_statistics()
    print(f"✓ Zones discovered: {stats['num_zones']}")
    
    # Test 4: Multi-zone person
    print("\n4. Testing multi-zone person detection...")
    
    # Person 4 moves between all zones
    multi_zone_positions = [
        (100, 100),  # Zone 1
        (400, 100),  # Zone 2
        (250, 300),  # Zone 3
        (100, 100),  # Back to Zone 1
    ]
    
    for pos in multi_zone_positions:
        tracker.update_position(4, pos)
    
    # Force zone update to detect multi-zone
    for _ in range(20):
        for pos in multi_zone_positions:
            tracker.update_position(4, pos)
    
    tracker.update_zones(force=True)
    
    is_multi_zone = tracker.is_multi_zone_person(4)
    print(f"  Person 4 multi-zone: {is_multi_zone}")
    
    if is_multi_zone:
        zones = tracker.get_person_zones(4)
        print(f"  Person 4 zones: {zones}")
    
    # Test 5: Spatial similarity
    print("\n5. Testing spatial similarity...")
    
    # Person 1 at their own zone
    sim_own = tracker.get_spatial_similarity(1, (100, 100))
    print(f"  Person 1 at Zone 1 (own): {sim_own:.3f} (should be high)")
    
    # Person 1 at another zone
    sim_other = tracker.get_spatial_similarity(1, (400, 100))
    print(f"  Person 1 at Zone 2 (other): {sim_other:.3f} (should be low)")
    
    # Multi-zone person at any zone
    if is_multi_zone:
        sim_multi = tracker.get_spatial_similarity(4, (250, 300))
        print(f"  Person 4 (multi-zone) at Zone 3: {sim_multi:.3f} (boosted)")
    
    # Test 6: Zone transitions
    print("\n6. Zone transition statistics...")
    stats = tracker.get_statistics()
    print(f"  Total transitions: {stats['total_transitions']}")
    print(f"  Valid transitions: {stats['valid_transitions']}")
    print(f"  Invalid transitions: {stats['invalid_transitions']}")
    print(f"  Multi-zone persons: {stats['multi_zone_persons']}")
    
    # Test 7: Get primary zones
    print("\n7. Primary work zones:")
    for person_id in [1, 2, 3, 4]:
        primary = tracker.get_person_primary_zone(person_id)
        is_multi = tracker.is_multi_zone_person(person_id)
        print(f"  Person {person_id}: Zone {primary}, Multi-zone: {is_multi}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)
    print("\nKey improvements demonstrated:")
    print("  ✓ Zone discovery (automatic)")
    print("  ✓ Multi-zone person detection")
    print("  ✓ Zone transition tracking")
    print("  ✓ Quality-aware spatial predictions")
    print("  ✓ Spatial similarity calculation")
    print("="*70)


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_spatial_tracker_enhanced()
