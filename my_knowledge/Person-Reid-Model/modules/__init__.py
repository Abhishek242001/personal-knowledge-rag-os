"""
modules/__init__.py - Enhanced Module Package Initialization
============================================================

Exports all modules for the Enhanced Multi-Camera Person ReID System v2.0

✅ NEW MODULES:
- adaptive_weights: Adaptive weight adjustment
- temporal_smoother: Temporal consistency
- occlusion_handler: Occlusion-robust features

✅ ENHANCED MODULES:
- global_reid_db: Hierarchical matching
- skeleton_reid: Occlusion integration
- spatial_tracker: Multi-zone tracking

Version: 2.0 (Enhanced)
Author: AI Team
Date: 2026-01-05
"""

# Version info
__version__ = "2.0"
__release__ = "Enhanced Multi-Camera ReID"
__author__ = "AI Team"
__date__ = "2026-01-05"

# Core detection and ReID modules
from .person_detector import PersonDetector
from .osnet_reid import OSNetFeatureExtractor

# ✅ ENHANCED: Skeleton ReID with occlusion handling
from .skeleton_reid import SkeletonFeatureExtractor, extract_skeleton_features

# ✅ NEW: Occlusion handler
from .occlusion_handler import OcclusionHandler

# ✅ ENHANCED: Global database with hierarchical matching
from .global_reid_db import GlobalReIDDatabase, PersonRecord

# ✅ ENHANCED: Spatial tracker with multi-zone detection
from .spatial_tracker import SpatialTracker

# ✅ NEW: Adaptive weights
from .adaptive_weights import AdaptiveWeightManager

# ✅ NEW: Temporal smoother
from .temporal_smoother import TemporalSmoother

# Output modules
from .json_writer import JSONWriter

# Configuration (from root directory)
# Note: config.py is in the project root, not in modules/
# Import it as a separate import when needed, e.g.:
# import config
# Or access via sys.path if needed

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Core modules
    'PersonDetector',
    'OSNetFeatureExtractor',
    'SkeletonFeatureExtractor',
    'extract_skeleton_features',
    
    # ✅ NEW: Enhancement modules
    'OcclusionHandler',
    'AdaptiveWeightManager',
    'TemporalSmoother',
    
    # Global tracking
    'GlobalReIDDatabase',
    'PersonRecord',
    'SpatialTracker',
    
    # Output
    'JSONWriter',
    
    # Version info
    '__version__',
    '__release__',
]


# ============================================================================
# FEATURE FLAGS
# ============================================================================

ENHANCEMENTS_AVAILABLE = {
    'hierarchical_matching': True,      # 3-pass matching strategy
    'occlusion_handling': True,         # Bayesian imputation
    'adaptive_weights': True,           # Context-aware weights
    'temporal_smoothing': True,         # Reduce jitter
    'multi_zone_tracking': True,        # Automatic multi-zone detection
}


# ============================================================================
# COMPATIBILITY CHECKS
# ============================================================================

def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        dict: Status of each dependency
    """
    dependencies = {}
    
    # Check NumPy
    try:
        import numpy
        dependencies['numpy'] = {
            'available': True,
            'version': numpy.__version__
        }
    except ImportError:
        dependencies['numpy'] = {'available': False, 'version': None}
    
    # Check OpenCV
    try:
        import cv2
        dependencies['opencv'] = {
            'available': True,
            'version': cv2.__version__
        }
    except ImportError:
        dependencies['opencv'] = {'available': False, 'version': None}
    
    # Check PyTorch
    try:
        import torch
        dependencies['torch'] = {
            'available': True,
            'version': torch.__version__,
            'cuda': torch.cuda.is_available()
        }
    except ImportError:
        dependencies['torch'] = {'available': False, 'version': None}
    
    # Check scikit-learn (for DBSCAN)
    try:
        import sklearn
        dependencies['sklearn'] = {
            'available': True,
            'version': sklearn.__version__
        }
    except ImportError:
        dependencies['sklearn'] = {'available': False, 'version': None}
    
    # Check ultralytics (for YOLO11)
    try:
        import ultralytics
        dependencies['ultralytics'] = {
            'available': True,
            'version': ultralytics.__version__
        }
    except ImportError:
        dependencies['ultralytics'] = {'available': False, 'version': None}
    
    return dependencies


def print_system_info():
    """Print system information and available features."""
    print("\n" + "="*70)
    print(f"ENHANCED MULTI-CAMERA REID SYSTEM v{__version__}")
    print(f"Release: {__release__}")
    print("="*70)
    
    print("\n📦 AVAILABLE ENHANCEMENTS:")
    for feature, available in ENHANCEMENTS_AVAILABLE.items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    
    print("\n📚 DEPENDENCIES:")
    deps = check_dependencies()
    
    for name, info in deps.items():
        if info['available']:
            version = info.get('version', 'unknown')
            status = "✅"
            extra = ""
            
            if name == 'torch' and 'cuda' in info:
                extra = f" (CUDA: {'✅' if info['cuda'] else '❌'})"
            
            print(f"  {status} {name}: v{version}{extra}")
        else:
            print(f"  ❌ {name}: NOT INSTALLED")
    
    # Check for critical missing dependencies
    critical = ['numpy', 'opencv', 'torch', 'sklearn', 'ultralytics']
    missing_critical = [name for name in critical if not deps.get(name, {}).get('available', False)]
    
    if missing_critical:
        print(f"\n⚠️  WARNING: Missing critical dependencies: {', '.join(missing_critical)}")
        print("   Install with: pip install -r requirements.txt")
    else:
        print("\n✅ All critical dependencies installed")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def initialize_system(validate_config: bool = True, verbose: bool = True):
    """
    Initialize the ReID system.
    
    Args:
        validate_config: Whether to validate configuration
        verbose: Whether to print system info
    
    Returns:
        bool: True if initialization successful
    """
    success = True
    
    if verbose:
        print_system_info()
    
    # Validate configuration (import from root)
    if validate_config:
        try:
            import sys
            from pathlib import Path
            
            # Add parent directory to path
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            from modules import config
            config.validate_config()
            if verbose:
                print("✅ Configuration validated")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            success = False
    
    # Check dependencies
    deps = check_dependencies()
    critical_missing = []
    
    for name in ['numpy', 'opencv', 'torch', 'sklearn', 'ultralytics']:
        if not deps.get(name, {}).get('available', False):
            critical_missing.append(name)
    
    if critical_missing:
        print(f"❌ Missing critical dependencies: {', '.join(critical_missing)}")
        success = False
    
    return success


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_system(camera_id: str = "cam1",
                 enable_adaptive_weights: bool = True,
                 enable_temporal_smoothing: bool = True,
                 enable_json_output: bool = True,
                 enable_visualization: bool = True):
    """
    Convenience function to create a fully configured ReID system.
    
    Args:
        camera_id: Camera identifier
        enable_adaptive_weights: Enable adaptive weight adjustment
        enable_temporal_smoothing: Enable temporal smoothing
        enable_json_output: Enable JSON output
        enable_visualization: Enable visualization
    
    Returns:
        Configured system instance
    """
    # Import from main module (in parent directory)
    import sys
    from pathlib import Path
    
    # Add parent directory to path if needed
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from main import MultiCameraReIDSystem
    
    return MultiCameraReIDSystem(
        camera_id=camera_id,
        enable_visualization=enable_visualization,
        enable_json_output=enable_json_output,
        enable_adaptive_weights=enable_adaptive_weights,
        enable_temporal_smoothing=enable_temporal_smoothing
    )


# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

# Print info when module is imported (can be disabled)
if __name__ != "__main__":
    # Only show brief version info on import
    import sys
    if '--verbose' in sys.argv or '--system-info' in sys.argv:
        print_system_info()


# ============================================================================
# TESTING INTERFACE
# ============================================================================

if __name__ == "__main__":
    """Run module tests when executed directly."""
    import sys
    
    print("\n" + "="*70)
    print("MODULE PACKAGE TEST")
    print("="*70)
    
    # Print system info
    print_system_info()
    
    # Test imports
    print("\n🧪 TESTING MODULE IMPORTS:")
    
    modules_to_test = [
        ('PersonDetector', PersonDetector),
        ('OSNetFeatureExtractor', OSNetFeatureExtractor),
        ('SkeletonFeatureExtractor', SkeletonFeatureExtractor),
        ('OcclusionHandler', OcclusionHandler),
        ('GlobalReIDDatabase', GlobalReIDDatabase),
        ('SpatialTracker', SpatialTracker),
        ('AdaptiveWeightManager', AdaptiveWeightManager),
        ('TemporalSmoother', TemporalSmoother),
        ('JSONWriter', JSONWriter),
    ]
    
    all_success = True
    for name, module in modules_to_test:
        try:
            # Test instantiation would go here
            print(f"  ✅ {name}: Available")
        except Exception as e:
            print(f"  ❌ {name}: Failed - {e}")
            all_success = False
    
    # Initialize system
    print("\n🚀 INITIALIZING SYSTEM:")
    success = initialize_system(validate_config=True, verbose=False)
    
    if success:
        print("  ✅ System initialization successful")
    else:
        print("  ❌ System initialization failed")
        all_success = False
    
    # Final result
    print("\n" + "="*70)
    if all_success:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    print("="*70 + "\n")
