#!/usr/bin/env python3
"""
NPZ to Text Converter
Convert skeleton NPZ files to human-readable text format for labeling and training
"""

import os
import sys
import argparse
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NPZToTextConverter:
    """Convert NPZ skeleton files to text format"""
    
    # H36M joint names (17 joints)
    JOINT_NAMES = [
        'Hip',           # 0
        'RHip',          # 1
        'RKnee',         # 2
        'RAnkle',        # 3
        'LHip',          # 4
        'LKnee',         # 5
        'LAnkle',        # 6
        'Spine',         # 7
        'Thorax',        # 8
        'Neck',          # 9
        'Head',          # 10
        'LShoulder',     # 11
        'LElbow',        # 12
        'LWrist',        # 13
        'RShoulder',     # 14
        'RElbow',        # 15
        'RWrist'         # 16
    ]
    
    def __init__(self, input_dir: str, output_dir: str, format_type: str = 'standard'):
        """
        Initialize converter
        
        Args:
            input_dir: Directory containing NPZ files
            output_dir: Directory for output text files
            format_type: Output format ('standard', 'compact', 'csv')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'converted': 0,
            'failed': 0,
            'total_frames': 0
        }
    
    def find_npz_files(self) -> List[Path]:
        """Find all NPZ files in input directory"""
        npz_files = list(self.input_dir.rglob('*.npz'))
        self.stats['total_files'] = len(npz_files)
        logger.info(f"Found {len(npz_files)} NPZ files")
        return npz_files
    
    def load_npz_data(self, npz_path: Path) -> Optional[np.ndarray]:
        """
        Load data from NPZ file
        
        Args:
            npz_path: Path to NPZ file
            
        Returns:
            Numpy array with skeleton data or None if failed
        """
        try:
            data = np.load(npz_path)
            
            if 'reconstruction' not in data:
                logger.error(f"No 'reconstruction' key in {npz_path}")
                return None
            
            reconstruction = data['reconstruction']
            data.close()
            
            logger.debug(f"Loaded data shape: {reconstruction.shape}")
            return reconstruction
            
        except Exception as e:
            logger.error(f"Failed to load {npz_path}: {str(e)}")
            return None
    
    def get_output_path(self, npz_path: Path, extension: str = '.txt') -> Path:
        """Generate output path maintaining directory structure"""
        relative_path = npz_path.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = relative_path.stem + extension
        return output_path / filename
    
    def write_standard_format(self, data: np.ndarray, output_file: Path, metadata: Dict):
        """
        Write data in standard human-readable format
        
        Format:
        - Header with metadata
        - Frame-by-frame joint coordinates
        - Joint names included
        """
        with open(output_file, 'w') as f:
            # Write header
            f.write("# SKELETON DATA FILE\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Source: {metadata['source']}\n")
            f.write(f"# Subject: {metadata.get('subject', 'Unknown')}\n")
            f.write(f"# Activity: {metadata.get('activity', 'Unknown')}\n")
            f.write("#" + "="*70 + "\n")
            
            # Handle different data shapes
            if len(data.shape) == 4:  # Multiple people
                num_people, num_frames, num_joints, _ = data.shape
                f.write(f"# Format: Multi-person ({num_people} people)\n")
                f.write(f"# Frames: {num_frames}\n")
                f.write(f"# Joints: {num_joints}\n")
                f.write("#" + "="*70 + "\n\n")
                
                for person_idx in range(num_people):
                    f.write(f"PERSON {person_idx + 1}\n")
                    f.write("-"*70 + "\n")
                    self._write_person_data(f, data[person_idx])
                    f.write("\n")
                    
            else:  # Single person
                num_frames, num_joints, _ = data.shape
                f.write(f"# Format: Single person\n")
                f.write(f"# Frames: {num_frames}\n")
                f.write(f"# Joints: {num_joints}\n")
                f.write("#" + "="*70 + "\n\n")
                self._write_person_data(f, data)
    
    def _write_person_data(self, file_handle, person_data: np.ndarray):
        """Write single person data to file"""
        num_frames, num_joints, _ = person_data.shape
        
        for frame_idx in range(num_frames):
            file_handle.write(f"Frame {frame_idx + 1:04d}\n")
            file_handle.write("Joint#  Joint_Name      X          Y          Z\n")
            file_handle.write("-"*60 + "\n")
            
            frame_data = person_data[frame_idx]
            for joint_idx in range(num_joints):
                joint_name = self.JOINT_NAMES[joint_idx] if joint_idx < len(self.JOINT_NAMES) else f"Joint{joint_idx}"
                x, y, z = frame_data[joint_idx]
                file_handle.write(f"{joint_idx:3d}     {joint_name:12s} {x:10.6f} {y:10.6f} {z:10.6f}\n")
            
            file_handle.write("\n")
    
    def write_compact_format(self, data: np.ndarray, output_file: Path, metadata: Dict):
        """
        Write data in compact format (space-efficient)
        
        Format:
        - Minimal header
        - Comma-separated values
        - No joint names
        """
        with open(output_file, 'w') as f:
            # Write minimal header
            f.write(f"# {metadata.get('source', 'Unknown')}\n")
            f.write(f"# Shape: {data.shape}\n")
            
            # Flatten and write data
            if len(data.shape) == 4:  # Multiple people
                for person_idx in range(data.shape[0]):
                    f.write(f"# Person {person_idx + 1}\n")
                    person_data = data[person_idx]
                    for frame_idx in range(person_data.shape[0]):
                        frame_flat = person_data[frame_idx].flatten()
                        f.write(','.join(map(str, frame_flat)) + '\n')
            else:  # Single person
                for frame_idx in range(data.shape[0]):
                    frame_flat = data[frame_idx].flatten()
                    f.write(','.join(map(str, frame_flat)) + '\n')
    
    def write_csv_format(self, data: np.ndarray, output_file: Path, metadata: Dict):
        """
        Write data in CSV format for easy import to spreadsheet/ML tools
        
        Format:
        - CSV header with column names
        - One row per frame
        - Columns: frame_id, person_id, joint_0_x, joint_0_y, joint_0_z, ...
        """
        import csv
        
        with open(output_file, 'w', newline='') as f:
            # Generate column headers
            headers = ['frame_id']
            
            if len(data.shape) == 4:  # Multiple people
                headers.append('person_id')
            
            # Add joint coordinate columns
            for joint_idx in range(data.shape[-2]):
                joint_name = self.JOINT_NAMES[joint_idx] if joint_idx < len(self.JOINT_NAMES) else f"joint_{joint_idx}"
                headers.extend([f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"])
            
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # Write data
            if len(data.shape) == 4:  # Multiple people
                for person_idx in range(data.shape[0]):
                    person_data = data[person_idx]
                    for frame_idx in range(person_data.shape[0]):
                        row = [frame_idx + 1, person_idx + 1]
                        frame_flat = person_data[frame_idx].flatten()
                        row.extend(frame_flat)
                        writer.writerow(row)
            else:  # Single person
                for frame_idx in range(data.shape[0]):
                    row = [frame_idx + 1]
                    frame_flat = data[frame_idx].flatten()
                    row.extend(frame_flat)
                    writer.writerow(row)
    
    def convert_file(self, npz_path: Path) -> bool:
        """
        Convert single NPZ file to text format
        
        Args:
            npz_path: Path to NPZ file
            
        Returns:
            Success status
        """
        logger.info(f"Converting: {npz_path}")
        
        # Load data
        data = self.load_npz_data(npz_path)
        if data is None:
            self.stats['failed'] += 1
            return False
        
        # Extract metadata from path
        relative_path = npz_path.relative_to(self.input_dir)
        parts = relative_path.parts
        
        metadata = {
            'source': str(npz_path.name),
            'subject': parts[1] if len(parts) > 1 else 'Unknown',
            'activity': parts[2] if len(parts) > 2 else 'Unknown'
        }
        
        # Determine output file and extension based on format
        extensions = {
            'standard': '.txt',
            'compact': '.compact.txt',
            'csv': '.csv'
        }
        
        output_file = self.get_output_path(npz_path, extensions.get(self.format_type, '.txt'))
        
        try:
            # Write based on format type
            if self.format_type == 'standard':
                self.write_standard_format(data, output_file, metadata)
            elif self.format_type == 'compact':
                self.write_compact_format(data, output_file, metadata)
            elif self.format_type == 'csv':
                self.write_csv_format(data, output_file, metadata)
            else:
                raise ValueError(f"Unknown format type: {self.format_type}")
            
            # Update statistics
            self.stats['converted'] += 1
            if len(data.shape) == 4:
                self.stats['total_frames'] += data.shape[0] * data.shape[1]
            else:
                self.stats['total_frames'] += data.shape[0]
            
            logger.info(f"✓ Converted to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {npz_path}: {str(e)}")
            self.stats['failed'] += 1
            return False
    
    def convert_all(self):
        """Convert all NPZ files in input directory"""
        npz_files = self.find_npz_files()
        
        if not npz_files:
            logger.warning("No NPZ files found")
            return
        
        for npz_file in npz_files:
            self.convert_file(npz_file)
        
        self.print_summary()
    
    def print_summary(self):
        """Print conversion summary"""
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print(f"Total files found: {self.stats['total_files']}")
        print(f"Successfully converted: {self.stats['converted']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Total frames processed: {self.stats['total_frames']}")
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Convert NPZ skeleton files to text format'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./output/GMDC5A24',
        help='Directory containing NPZ files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/GMDC5A24_text',
        help='Directory for output text files'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['standard', 'compact', 'csv'],
        default='standard',
        help='Output format type'
    )
    parser.add_argument(
        '--single-file',
        type=str,
        help='Convert single NPZ file instead of batch'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = NPZToTextConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        format_type=args.format
    )
    
    # Convert files
    if args.single_file:
        npz_path = Path(args.single_file)
        if not npz_path.exists():
            logger.error(f"File not found: {npz_path}")
            sys.exit(1)
        converter.convert_file(npz_path)
    else:
        converter.convert_all()

if __name__ == "__main__":
    main()
