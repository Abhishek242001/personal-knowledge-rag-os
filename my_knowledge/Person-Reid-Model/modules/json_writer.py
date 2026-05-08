"""
json_writer.py - JSON Response Writer
======================================

Utility for writing JSON responses to disk with proper formatting.

Author: AI Team
Date: 2024-12-31
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class JSONWriter:
    """
    JSON response writer with automatic directory management.
    
    Features:
    - Automatic directory creation
    - Frame-numbered filenames
    - Pretty-printed JSON
    - Error handling
    - Optional summary file
    """
    
    def __init__(self, output_dir: str = "output/json", create_summary: bool = False):
        """
        Initialize JSON writer.
        
        Args:
            output_dir: Directory for JSON output files
            create_summary: Create summary file with all responses
        """
        self.output_dir = Path(output_dir)
        self.create_summary = create_summary
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JSON writer initialized: {self.output_dir}")
        
        # Summary file
        if create_summary:
            self.summary_path = self.output_dir / "summary.jsonl"
            self.summary_file = open(self.summary_path, 'w', encoding='utf-8')
            logger.info(f"Summary file: {self.summary_path}")
        else:
            self.summary_file = None
        
        self.frame_count = 0
        self.total_written = 0
        self.errors = 0
    
    def write_frame_response(self, frame_number: int, response: Dict[str, Any]) -> bool:
        """
        Write frame response to JSON file.
        
        Args:
            frame_number: Frame number
            response: Response dictionary
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate filename
            filename = self.output_dir / f"frame_{frame_number:06d}.json"
            
            # Add metadata
            response['_metadata'] = {
                'frame_number': frame_number,
                'timestamp': datetime.now().isoformat(),
                'written_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Write JSON file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            
            # Write to summary file
            if self.summary_file:
                self.summary_file.write(json.dumps(response, ensure_ascii=False) + '\n')
                self.summary_file.flush()
            
            self.frame_count += 1
            self.total_written += 1
            
            if self.frame_count % 100 == 0:
                logger.info(f"Written {self.frame_count} JSON files")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write JSON for frame {frame_number}: {e}")
            self.errors += 1
            return False
    
    def write_custom(self, filename: str, data: Dict[str, Any]) -> bool:
        """
        Write custom JSON file.
        
        Args:
            filename: Output filename (without directory)
            data: Data dictionary
        
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Written custom JSON: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write custom JSON {filename}: {e}")
            return False
    
    def write_statistics(self, stats: Dict[str, Any], filename: str = "statistics.json"):
        """
        Write statistics to JSON file.
        
        Args:
            stats: Statistics dictionary
            filename: Output filename
        """
        stats['_metadata'] = {
            'total_frames_written': self.frame_count,
            'total_files_written': self.total_written,
            'errors': self.errors,
            'generated_at': datetime.now().isoformat()
        }
        
        self.write_custom(filename, stats)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get writer statistics."""
        return {
            'frames_written': self.frame_count,
            'total_files': self.total_written,
            'errors': self.errors,
            'output_directory': str(self.output_dir),
            'summary_enabled': self.create_summary
        }
    
    def close(self):
        """Close writer and summary file."""
        if self.summary_file:
            self.summary_file.close()
            logger.info(f"Summary file closed: {self.summary_path}")
        
        logger.info(f"JSON writer closed. Total written: {self.total_written}, Errors: {self.errors}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def test_json_writer():
    """Test JSON writer."""
    import sys
    from pathlib import Path
    
    print("\n" + "="*70)
    print("JSON WRITER - TEST")
    print("="*70)
    
    # Create test output directory
    test_output = Path("/home/claude/test_json_output")
    test_output.mkdir(parents=True, exist_ok=True)
    
    # Initialize writer
    print(f"\nInitializing JSON writer...")
    writer = JSONWriter(output_dir=str(test_output), create_summary=True)
    print(f"✓ Writer initialized: {test_output}")
    
    # Write test frames
    print("\nWriting test frames...")
    for frame_num in range(1, 6):
        response = {
            'frame_count': frame_num,
            'total_people_detected': 3,
            'detection': [
                {'person_id': 1, 'confidence': 0.95},
                {'person_id': 2, 'confidence': 0.87},
                {'person_id': 3, 'confidence': 0.92}
            ]
        }
        
        success = writer.write_frame_response(frame_num, response)
        if success:
            print(f"  ✓ Frame {frame_num}")
        else:
            print(f"  ✗ Frame {frame_num} failed")
    
    # Write statistics
    print("\nWriting statistics...")
    stats = {
        'total_frames': 5,
        'total_people': 3,
        'processing_time_ms': 1234.5
    }
    writer.write_statistics(stats)
    print("✓ Statistics written")
    
    # Get writer stats
    print("\nWriter statistics:")
    writer_stats = writer.get_statistics()
    for key, value in writer_stats.items():
        print(f"  {key}: {value}")
    
    # Close writer
    print("\nClosing writer...")
    writer.close()
    print("✓ Writer closed")
    
    # Verify files
    print("\nVerifying output files...")
    json_files = list(test_output.glob("*.json"))
    print(f"  JSON files: {len(json_files)}")
    
    jsonl_files = list(test_output.glob("*.jsonl"))
    print(f"  JSONL files: {len(jsonl_files)}")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_json_writer()
