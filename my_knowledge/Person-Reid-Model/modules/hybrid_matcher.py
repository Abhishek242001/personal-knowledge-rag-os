"""
hybrid_matcher.py - Multi-Modal Fusion Engine
==============================================

Combines three modalities for robust person re-identification:
1. Skeleton features (50% weight) - Body measurements
2. Spatial features (30% weight) - Work zone patterns
3. Appearance features (20% weight) - OSNet embeddings

This fusion strategy enables high accuracy even with identical uniforms.

Author: AI Team
Date: 2024-12-31
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from . import config

logger = logging.getLogger(__name__)


@dataclass
class MultiModalFeatures:
    """Container for multi-modal features."""
    skeleton_features: Optional[np.ndarray] = None
    appearance_features: Optional[np.ndarray] = None
    position: Optional[Tuple[float, float]] = None
    timestamp: Optional[float] = None
    
    def is_complete(self) -> bool:
        """Check if all modalities are present."""
        return (self.skeleton_features is not None and
                self.appearance_features is not None and
                self.position is not None)
    
    def available_modalities(self) -> List[str]:
        """Get list of available modalities."""
        modalities = []
        if self.skeleton_features is not None:
            modalities.append('skeleton')
        if self.appearance_features is not None:
            modalities.append('appearance')
        if self.position is not None:
            modalities.append('spatial')
        return modalities


class HybridMatcher:
    """
    Multi-modal fusion engine for person re-identification.
    
    Combines skeleton, spatial, and appearance features using weighted fusion.
    Adapts weights based on feature availability and quality.
    """
    
    def __init__(self,
                 skeleton_weight: float = None,
                 spatial_weight: float = None,
                 appearance_weight: float = None,
                 similarity_threshold: float = None,
                 adaptive_weights: bool = None,
                 min_modalities: int = None):
        """
        Initialize hybrid matcher.
        
        Args:
            skeleton_weight: Weight for skeleton features (default: 0.50)
            spatial_weight: Weight for spatial features (default: 0.30)
            appearance_weight: Weight for appearance features (default: 0.20)
            similarity_threshold: Minimum similarity for match (0.0-1.0)
            adaptive_weights: Enable adaptive weight adjustment
            min_modalities: Minimum modalities required for matching
        """
        
        # Use config defaults if not specified
        if skeleton_weight is None:
            skeleton_weight = config.FUSION.SKELETON_WEIGHT
        if spatial_weight is None:
            spatial_weight = config.FUSION.SPATIAL_WEIGHT
        if appearance_weight is None:
            appearance_weight = config.FUSION.APPEARANCE_WEIGHT
        if similarity_threshold is None:
            similarity_threshold = config.FUSION.SIMILARITY_THRESHOLD
        if adaptive_weights is None:
            adaptive_weights = config.FUSION.ADAPTIVE_WEIGHTS
        if min_modalities is None:
            min_modalities = config.FUSION.MIN_MODALITIES

        # Validate weights
        total = skeleton_weight + spatial_weight + appearance_weight
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.skeleton_weight = skeleton_weight
        self.spatial_weight = spatial_weight
        self.appearance_weight = appearance_weight
        self.similarity_threshold = similarity_threshold
        self.adaptive_weights = adaptive_weights
        self.min_modalities = min_modalities
        
        # Statistics
        self.total_matches = 0
        self.skeleton_matches = 0
        self.spatial_matches = 0
        self.appearance_matches = 0
        self.fusion_matches = 0
        
        logger.info("Hybrid Matcher initialized")
        logger.info(f"  Weights: Skeleton={skeleton_weight:.0%}, "
                   f"Spatial={spatial_weight:.0%}, "
                   f"Appearance={appearance_weight:.0%}")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Adaptive weights: {adaptive_weights}")
    
    def calculate_similarity(self,
                           features1: MultiModalFeatures,
                           features2: MultiModalFeatures,
                           skeleton_extractor=None,
                           spatial_tracker=None,
                           appearance_extractor=None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-modal similarity between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            skeleton_extractor: Skeleton feature extractor (for similarity calculation)
            spatial_tracker: Spatial tracker (for similarity calculation)
            appearance_extractor: Appearance feature extractor (for similarity calculation)
        
        Returns:
            Tuple of (final_similarity, individual_similarities_dict)
        """
        # Check available modalities
        modalities1 = set(features1.available_modalities())
        modalities2 = set(features2.available_modalities())
        common_modalities = modalities1.intersection(modalities2)
        
        if len(common_modalities) < self.min_modalities:
            logger.debug(f"Insufficient common modalities: {len(common_modalities)} < {self.min_modalities}")
            return 0.0, {}
        
        # Calculate individual similarities
        similarities = {}
        weights = {}
        
        # 1. Skeleton similarity
        if ('skeleton' in common_modalities and 
            skeleton_extractor is not None):
            skeleton_sim = skeleton_extractor.calculate_similarity(
                features1.skeleton_features,
                features2.skeleton_features
            )
            similarities['skeleton'] = skeleton_sim
            weights['skeleton'] = self.skeleton_weight
        
        # 2. Spatial similarity
        # Note: For spatial, we typically compare person's historical pattern
        # with current position. This is handled differently.
        if 'spatial' in common_modalities:
            # Calculate spatial consistency
            # (in practice, this is usually done via spatial_tracker.get_spatial_similarity)
            # For direct comparison, calculate position distance
            pos1 = np.array(features1.position)
            pos2 = np.array(features2.position)
            distance = np.linalg.norm(pos1 - pos2)
            
            # Convert distance to similarity (exponential decay)
            spatial_sim = np.exp(-distance / config.SPATIAL.DISTANCE_SCALE)
            similarities['spatial'] = float(spatial_sim)
            weights['spatial'] = self.spatial_weight
        
        # 3. Appearance similarity
        if ('appearance' in common_modalities and 
            appearance_extractor is not None):
            appearance_sim = self._calculate_cosine_similarity(
                features1.appearance_features,
                features2.appearance_features
            )
            similarities['appearance'] = appearance_sim
            weights['appearance'] = self.appearance_weight
        
        # Adaptive weight adjustment
        if self.adaptive_weights:
            weights = self._adjust_weights_adaptive(weights, similarities)
        else:
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted fusion
        final_similarity = 0.0
        for modality, sim in similarities.items():
            weight = weights.get(modality, 0.0)
            final_similarity += weight * sim
            logger.debug(f"  {modality}: sim={sim:.3f}, weight={weight:.3f}")
        
        logger.debug(f"Final similarity: {final_similarity:.3f}")
        
        return final_similarity, similarities
    
    def _adjust_weights_adaptive(self,
                                weights: Dict[str, float],
                                similarities: Dict[str, float]) -> Dict[str, float]:
        """
        Adaptively adjust weights based on individual similarities.
        
        Strategy: Boost weights of high-confidence modalities.
        
        Args:
            weights: Initial weights
            similarities: Individual similarities
        
        Returns:
            Adjusted weights (normalized to sum to 1.0)
        """
        adjusted_weights = {}
        
        for modality, base_weight in weights.items():
            if modality not in similarities:
                adjusted_weights[modality] = base_weight
                continue
            
            sim = similarities[modality]
            
            # Confidence factor: boost if very high (>0.8) or very low (<0.3)
            # High similarity = high confidence = boost
            # Low similarity = clear mismatch = also valuable signal
            if sim > config.FUSION.CONFIDENCE_BOOST_HIGH:
                confidence = 1.2  # Boost by 20%
            elif sim < config.FUSION.CONFIDENCE_BOOST_LOW:
                confidence = 1.1  # Boost by 10%
            else:
                confidence = 1.0  # Keep as-is
            
            adjusted_weights[modality] = base_weight * confidence
        
        # Normalize to sum to 1.0
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _calculate_cosine_similarity(self,
                                    features1: np.ndarray,
                                    features2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            
            # Convert from [-1, 1] to [0, 1]
            similarity = (cosine_sim + 1.0) / 2.0
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def find_best_match(self,
                       query_features: MultiModalFeatures,
                       candidate_features: Dict[int, MultiModalFeatures],
                       skeleton_extractor=None,
                       spatial_tracker=None,
                       appearance_extractor=None,
                       person_id_hint: Optional[int] = None) -> Optional[Tuple[int, float, Dict[str, float]]]:
        """
        Find best matching person from candidates.
        
        Args:
            query_features: Query person features
            candidate_features: Dictionary of {person_id: features}
            skeleton_extractor: Skeleton feature extractor
            spatial_tracker: Spatial tracker
            appearance_extractor: Appearance feature extractor
            person_id_hint: Optional person ID hint (from spatial prior)
        
        Returns:
            Tuple of (person_id, similarity, individual_similarities) or None
        """
        if not candidate_features:
            return None
        
        best_match_id = None
        best_similarity = 0.0
        best_individual_sims = {}
        
        # If we have a person ID hint from spatial prior, try it first
        if person_id_hint is not None and person_id_hint in candidate_features:
            sim, ind_sims = self.calculate_similarity(
                query_features,
                candidate_features[person_id_hint],
                skeleton_extractor,
                spatial_tracker,
                appearance_extractor
            )
            
            # Apply spatial bonus for hinted person
            spatial_bonus = config.SPATIAL.ZONE_MATCH_BONUS
            sim = min(1.0, sim + spatial_bonus)
            
            if sim >= self.similarity_threshold:
                logger.info(f"Spatial hint matched: person {person_id_hint} (sim={sim:.3f})")
                return person_id_hint, sim, ind_sims
        
        # Search all candidates
        for person_id, candidate_feat in candidate_features.items():
            sim, ind_sims = self.calculate_similarity(
                query_features,
                candidate_feat,
                skeleton_extractor,
                spatial_tracker,
                appearance_extractor
            )
            
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = person_id
                best_individual_sims = ind_sims
        
        # Check threshold
        if best_similarity >= self.similarity_threshold:
            self.total_matches += 1
            
            # Track which modality was strongest
            if best_individual_sims:
                strongest_modality = max(best_individual_sims.items(), key=lambda x: x[1])[0]
                if strongest_modality == 'skeleton':
                    self.skeleton_matches += 1
                elif strongest_modality == 'spatial':
                    self.spatial_matches += 1
                elif strongest_modality == 'appearance':
                    self.appearance_matches += 1
            
            self.fusion_matches += 1
            
            return best_match_id, best_similarity, best_individual_sims
        
        return None
    
    def evaluate_match_quality(self,
                              individual_similarities: Dict[str, float]) -> str:
        """
        Evaluate match quality based on individual similarities.
        
        Args:
            individual_similarities: Dictionary of individual modality similarities
        
        Returns:
            Quality string: 'excellent', 'good', 'fair', 'poor'
        """
        if not individual_similarities:
            return 'poor'
        
        avg_sim = np.mean(list(individual_similarities.values()))
        
        # Count high-confidence modalities (>0.7)
        high_confidence = sum(1 for sim in individual_similarities.values() if sim > config.FUSION.QUALITY_GOOD)
        
        if avg_sim > config.FUSION.QUALITY_EXCELLENT and high_confidence >= 2:
            return 'excellent'
        elif avg_sim > config.FUSION.QUALITY_GOOD or high_confidence >= 2:
            return 'good'
        elif avg_sim > config.FUSION.QUALITY_FAIR or high_confidence >= 1:
            return 'fair'
        else:
            return 'poor'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matching statistics."""
        stats = {
            'total_matches': self.total_matches,
            'fusion_matches': self.fusion_matches,
            'modality_breakdown': {
                'skeleton_strongest': self.skeleton_matches,
                'spatial_strongest': self.spatial_matches,
                'appearance_strongest': self.appearance_matches
            }
        }
        
        # Calculate percentages
        if self.total_matches > 0:
            stats['modality_percentages'] = {
                'skeleton': f"{self.skeleton_matches / self.total_matches * 100:.1f}%",
                'spatial': f"{self.spatial_matches / self.total_matches * 100:.1f}%",
                'appearance': f"{self.appearance_matches / self.total_matches * 100:.1f}%"
            }
        
        return stats
    
    def reset_statistics(self):
        """Reset matching statistics."""
        self.total_matches = 0
        self.skeleton_matches = 0
        self.spatial_matches = 0
        self.appearance_matches = 0
        self.fusion_matches = 0


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_hybrid_matcher():
    """Test the hybrid matcher."""
    import sys
    
    print("\n" + "="*70)
    print("HYBRID MATCHER - TEST")
    print("="*70)
    
    # Initialize matcher
    print("\nInitializing hybrid matcher...")
    try:
        matcher = HybridMatcher(
            skeleton_weight=0.50,
            spatial_weight=0.30,
            appearance_weight=0.20,
            similarity_threshold=0.60,
            adaptive_weights=True
        )
        print("✓ Hybrid matcher initialized")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        sys.exit(1)
    
    # Create test features
    print("\nCreating test feature sets...")
    
    # Person 1 features
    person1_features = MultiModalFeatures(
        skeleton_features=np.random.randn(128).astype(np.float32),
        appearance_features=np.random.randn(128).astype(np.float32),
        position=(100.0, 200.0),
        timestamp=time.time()
    )
    person1_features.skeleton_features /= np.linalg.norm(person1_features.skeleton_features)
    person1_features.appearance_features /= np.linalg.norm(person1_features.appearance_features)
    
    # Person 2 features (similar to person 1)
    person2_features = MultiModalFeatures(
        skeleton_features=person1_features.skeleton_features + np.random.randn(128) * 0.1,
        appearance_features=person1_features.appearance_features + np.random.randn(128) * 0.1,
        position=(105.0, 205.0),
        timestamp=time.time()
    )
    person2_features.skeleton_features /= np.linalg.norm(person2_features.skeleton_features)
    person2_features.appearance_features /= np.linalg.norm(person2_features.appearance_features)
    
    # Person 3 features (different from person 1)
    person3_features = MultiModalFeatures(
        skeleton_features=np.random.randn(128).astype(np.float32),
        appearance_features=np.random.randn(128).astype(np.float32),
        position=(300.0, 400.0),
        timestamp=time.time()
    )
    person3_features.skeleton_features /= np.linalg.norm(person3_features.skeleton_features)
    person3_features.appearance_features /= np.linalg.norm(person3_features.appearance_features)
    
    print("✓ Test features created")
    
    # Test similarity calculations
    print("\nTesting similarity calculations...")
    
    # Mock extractors (use simple cosine similarity)
    class MockExtractor:
        def calculate_similarity(self, f1, f2):
            dot = np.dot(f1, f2)
            return (dot + 1.0) / 2.0
    
    mock_extractor = MockExtractor()
    
    # Similar persons (should have high similarity)
    sim12, ind_sims12 = matcher.calculate_similarity(
        person1_features,
        person2_features,
        skeleton_extractor=mock_extractor,
        appearance_extractor=mock_extractor
    )
    print(f"  Person 1 vs Person 2 (similar): {sim12:.3f}")
    print(f"    Individual: {ind_sims12}")
    
    # Different persons (should have low similarity)
    sim13, ind_sims13 = matcher.calculate_similarity(
        person1_features,
        person3_features,
        skeleton_extractor=mock_extractor,
        appearance_extractor=mock_extractor
    )
    print(f"  Person 1 vs Person 3 (different): {sim13:.3f}")
    print(f"    Individual: {ind_sims13}")
    
    # Test match quality
    print("\nTesting match quality evaluation...")
    quality12 = matcher.evaluate_match_quality(ind_sims12)
    quality13 = matcher.evaluate_match_quality(ind_sims13)
    print(f"  Person 1 vs 2: {quality12}")
    print(f"  Person 1 vs 3: {quality13}")
    
    # Test find best match
    print("\nTesting best match finding...")
    candidates = {
        2: person2_features,
        3: person3_features
    }
    
    best_match = matcher.find_best_match(
        person1_features,
        candidates,
        skeleton_extractor=mock_extractor,
        appearance_extractor=mock_extractor
    )
    
    if best_match:
        match_id, match_sim, match_ind = best_match
        print(f"✓ Best match found: Person {match_id} (similarity={match_sim:.3f})")
    else:
        print("✗ No match found above threshold")
    
    # Statistics
    print("\nMatcher statistics:")
    stats = matcher.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_hybrid_matcher()
