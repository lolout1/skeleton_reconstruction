#!/usr/bin/env python3
"""
NPZ to CSV Converter
Convert skeleton NPZ files to CSV format where each frame is a row and each joint coordinate is a column
Compatible with both legacy directory structure and new standardized naming convention
"""

import os
import sys
import argparse
import numpy as np
import logging
import re
import csv
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

class StandardizedNameParser:
    """Parse standardized filenames to extract metadata"""
    
    def __init__(self):
        # Pattern for standardized names: S01_A01_T06.npz
        self.pattern = re.compile(r'S(\d+)_A(\d+)_T(\d+)\.npz$')
        
        # Activity code mapping
        self.activity_map = {
            '01': 'ADL',
            '10': 'Fall'
        }
    
    def parse(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Parse standardized filename to extract metadata
        
        Args:
            filename: Filename like 'S01_A01_T06.npz'
            
        Returns:
            Dictionary with subject, activity, trial info or None if no match
        """
        match = self.pattern.match(filename)
        if not match:
            return None
        
        subject_num, activity_code, trial_num = match.groups()
        
        return {
            'subject': f"Subject_{int(subject_num):02d}",
            'subject_id': f"S{int(subject_num):02d}",
            'activity': self.activity_map.get(activity_code, f"Activity_{activity_code}"),
            'activity_code': f"A{activity_code}",
            'trial': f"T{int(trial_num):02d}",
            'trial_num': int(trial_num),
            'original_filename': filename
        }

class NPZToCSVConverter:
    """Convert NPZ skeleton files to CSV format with frames as rows and joints as columns"""

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

    def __init__(self, input_dir: str, output_dir: str, include_metadata: bool = True, 
                 preserve_structure: bool = True):
        """
        Initialize converter

        Args:
            input_dir: Directory containing NPZ files
            output_dir: Directory for output CSV files
            include_metadata: Whether to include subject/activity/trial columns
            preserve_structure: Whether to preserve directory structure in output
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.include_metadata = include_metadata
        self.preserve_structure = preserve_structure
        
        # Initialize filename parser
        self.name_parser = StandardizedNameParser()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'total_files': 0,
            'converted': 0,
            'failed': 0,
            'total_frames': 0,
            'standardized_names': 0,
            'legacy_names': 0
        }

    def find_npz_files(self) -> List[Path]:
        """Find all NPZ files in input directory"""
        npz_files = []
        
        # Look for NPZ files recursively
        for npz_file in self.input_dir.rglob('*.npz'):
            # Skip non-skeleton files (like processing stats)
            if 'stats' not in npz_file.name.lower():
                npz_files.append(npz_file)
        
        # Also look for NPZ files directly in input directory
        for npz_file in self.input_dir.glob('*.npz'):
            if npz_file not in npz_files and 'stats' not in npz_file.name.lower():
                npz_files.append(npz_file)
        
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

    def extract_metadata(self, npz_path: Path) -> Dict[str, str]:
        """
        Extract metadata from NPZ file path/name
        
        Args:
            npz_path: Path to NPZ file
            
        Returns:
            Dictionary with metadata
        """
        filename = npz_path.name
        
        # Try to parse standardized filename first
        standardized_metadata = self.name_parser.parse(filename)
        
        if standardized_metadata:
            self.stats['standardized_names'] += 1
            logger.debug(f"Parsed standardized filename: {filename}")
            return {
                'source': filename,
                'subject': standardized_metadata['subject'],
                'subject_id': standardized_metadata['subject_id'],
                'activity': standardized_metadata['activity'],
                'activity_code': standardized_metadata['activity_code'],
                'trial': standardized_metadata['trial'],
                'trial_num': standardized_metadata['trial_num'],
                'naming_type': 'standardized'
            }
        else:
            # Fallback to legacy path-based extraction
            self.stats['legacy_names'] += 1
            logger.debug(f"Using legacy path parsing for: {filename}")
            
            relative_path = npz_path.relative_to(self.input_dir)
            parts = relative_path.parts
            
            # Try to extract from directory structure
            subject = 'Unknown'
            activity = 'Unknown'
            
            for part in parts:
                if 'Subject' in part or 'subject' in part:
                    subject = part.replace(' ', '_')
                elif part in ['ADL', 'Fall', 'adl', 'fall']:
                    activity = part
            
            return {
                'source': filename,
                'subject': subject,
                'subject_id': subject,
                'activity': activity,
                'activity_code': activity,
                'trial': npz_path.stem,
                'trial_num': 0,
                'naming_type': 'legacy'
            }

    def get_output_path(self, npz_path: Path, metadata: Dict[str, str]) -> Path:
        """
        Generate output path for CSV file
        
        Args:
            npz_path: Original NPZ file path
            metadata: Extracted metadata
            
        Returns:
            Output CSV file path
        """
        if metadata['naming_type'] == 'standardized':
            # For standardized names, create organized structure
            if self.preserve_structure:
                # Create Subject/Activity structure
                subject_dir = self.output_dir / metadata['subject_id']
                activity_dir = subject_dir / metadata['activity']
                activity_dir.mkdir(parents=True, exist_ok=True)
                
                # Use original standardized name
                output_filename = npz_path.stem + '.csv'
                return activity_dir / output_filename
            else:
                # Flat structure with standardized names
                output_filename = npz_path.stem + '.csv'
                return self.output_dir / output_filename
        else:
            # For legacy names, preserve original structure
            relative_path = npz_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = relative_path.stem + '.csv'
            return output_path / filename

    def generate_column_headers(self, num_joints: int, include_metadata: bool = True) -> List[str]:
        """
        Generate column headers for CSV file
        
        Args:
            num_joints: Number of joints in the data
            include_metadata: Whether to include metadata columns
            
        Returns:
            List of column header names
        """
        headers = []
        
        # Add metadata columns if requested
        if include_metadata:
            headers.extend(['Frame_ID', 'Subject', 'Activity', 'Trial'])
        else:
            headers.append('Frame_ID')
        
        # Add joint coordinate columns
        for joint_idx in range(num_joints):
            joint_name = self.JOINT_NAMES[joint_idx] if joint_idx < len(self.JOINT_NAMES) else f"Joint_{joint_idx:02d}"
            headers.extend([
                f"{joint_name}_X",
                f"{joint_name}_Y", 
                f"{joint_name}_Z"
            ])
        
        return headers

    def write_csv_format(self, data: np.ndarray, output_file: Path, metadata: Dict):
        """
        Write data in CSV format where each frame is a row and each joint coordinate is a column
        
        Args:
            data: Skeleton data array
            output_file: Output CSV file path
            metadata: File metadata
        """
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Handle different data shapes
            if len(data.shape) == 4:  # Multiple people
                num_people, num_frames, num_joints, coords = data.shape
                
                # Generate headers for multi-person data
                headers = []
                if self.include_metadata:
                    headers.extend(['Frame_ID', 'Person_ID', 'Subject', 'Activity', 'Trial'])
                else:
                    headers.extend(['Frame_ID', 'Person_ID'])
                
                # Add joint coordinate columns
                for joint_idx in range(num_joints):
                    joint_name = self.JOINT_NAMES[joint_idx] if joint_idx < len(self.JOINT_NAMES) else f"Joint_{joint_idx:02d}"
                    headers.extend([f"{joint_name}_X", f"{joint_name}_Y", f"{joint_name}_Z"])
                
                writer.writerow(headers)
                
                # Write data rows
                for person_idx in range(num_people):
                    person_data = data[person_idx]
                    for frame_idx in range(num_frames):
                        row = [frame_idx + 1, person_idx + 1]
                        
                        if self.include_metadata:
                            row.extend([metadata['subject'], metadata['activity'], metadata['trial']])
                        
                        # Add joint coordinates
                        frame_data = person_data[frame_idx]
                        for joint_idx in range(num_joints):
                            x, y, z = frame_data[joint_idx]
                            row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                        
                        writer.writerow(row)
                        
            else:  # Single person
                # Handle different array shapes
                if len(data.shape) == 3:
                    num_frames, num_joints, coords = data.shape
                else:
                    # Reshape if needed
                    data = data.reshape(-1, data.shape[-2], data.shape[-1])
                    num_frames, num_joints, coords = data.shape
                
                # Generate headers for single-person data
                headers = self.generate_column_headers(num_joints, self.include_metadata)
                writer.writerow(headers)
                
                # Write data rows
                for frame_idx in range(num_frames):
                    row = [frame_idx + 1]
                    
                    if self.include_metadata:
                        row.extend([metadata['subject'], metadata['activity'], metadata['trial']])
                    
                    # Add joint coordinates
                    frame_data = data[frame_idx]
                    for joint_idx in range(num_joints):
                        x, y, z = frame_data[joint_idx]
                        row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                    
                    writer.writerow(row)

    def convert_file(self, npz_path: Path) -> bool:
        """
        Convert single NPZ file to CSV format

        Args:
            npz_path: Path to NPZ file

        Returns:
            Success status
        """
        logger.info(f"Converting: {npz_path.name}")

        # Load data
        data = self.load_npz_data(npz_path)
        if data is None:
            self.stats['failed'] += 1
            return False

        # Extract metadata
        metadata = self.extract_metadata(npz_path)

        # Get output file path
        output_file = self.get_output_path(npz_path, metadata)

        try:
            # Write CSV format
            self.write_csv_format(data, output_file, metadata)

            # Update statistics
            self.stats['converted'] += 1
            if len(data.shape) == 4:
                self.stats['total_frames'] += data.shape[0] * data.shape[1]
            elif len(data.shape) == 3:
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

        logger.info(f"Starting conversion of {len(npz_files)} files...")
        
        for i, npz_file in enumerate(npz_files, 1):
            logger.info(f"[{i}/{len(npz_files)}] Processing...")
            self.convert_file(npz_file)

        self.print_summary()

    def create_combined_csv(self, npz_files: List[Path], output_file: Path):
        """
        Create a single combined CSV file with all data
        
        Args:
            npz_files: List of NPZ files to process
            output_file: Output combined CSV file path
        """
        logger.info(f"Creating combined CSV: {output_file}")
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers_written = False
            
            for npz_file in npz_files:
                logger.info(f"Adding to combined CSV: {npz_file.name}")
                
                # Load data
                data = self.load_npz_data(npz_file)
                if data is None:
                    continue
                
                # Extract metadata
                metadata = self.extract_metadata(npz_file)
                
                # Handle data shape
                if len(data.shape) == 4:  # Multiple people
                    num_people, num_frames, num_joints, coords = data.shape
                elif len(data.shape) == 3:  # Single person
                    data = data[np.newaxis, :]  # Add person dimension
                    num_people, num_frames, num_joints, coords = data.shape
                else:
                    continue
                
                # Write headers only once
                if not headers_written:
                    headers = ['File_Source', 'Frame_ID', 'Person_ID', 'Subject', 'Activity', 'Trial']
                    for joint_idx in range(num_joints):
                        joint_name = self.JOINT_NAMES[joint_idx] if joint_idx < len(self.JOINT_NAMES) else f"Joint_{joint_idx:02d}"
                        headers.extend([f"{joint_name}_X", f"{joint_name}_Y", f"{joint_name}_Z"])
                    writer.writerow(headers)
                    headers_written = True
                
                # Write data rows
                for person_idx in range(num_people):
                    person_data = data[person_idx]
                    for frame_idx in range(num_frames):
                        row = [
                            metadata['source'],
                            frame_idx + 1,
                            person_idx + 1,
                            metadata['subject'],
                            metadata['activity'],
                            metadata['trial']
                        ]
                        
                        # Add joint coordinates
                        frame_data = person_data[frame_idx]
                        for joint_idx in range(num_joints):
                            x, y, z = frame_data[joint_idx]
                            row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                        
                        writer.writerow(row)
        
        logger.info(f"✓ Combined CSV created: {output_file}")

    def print_summary(self):
        """Print conversion summary"""
        print("\n" + "="*70)
        print("CSV CONVERSION SUMMARY")
        print("="*70)
        print(f"Total files found:       {self.stats['total_files']}")
        print(f"Successfully converted:  {self.stats['converted']}")
        print(f"Failed:                  {self.stats['failed']}")
        print(f"Total frames processed:  {self.stats['total_frames']}")
        print(f"Standardized filenames:  {self.stats['standardized_names']}")
        print(f"Legacy filenames:        {self.stats['legacy_names']}")
        print(f"Include metadata:        {self.include_metadata}")
        print(f"Preserve structure:      {self.preserve_structure}")
        print("="*70)

def main():
    """Main function with enhanced options"""
    parser = argparse.ArgumentParser(
        description='Convert NPZ skeleton files to CSV format (frames as rows, joints as columns)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./output',
        help='Directory containing NPZ files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/csv_files',
        help='Directory for output CSV files'
    )
    parser.add_argument(
        '--single-file',
        type=str,
        help='Convert single NPZ file instead of batch'
    )
    parser.add_argument(
        '--no-structure',
        action='store_true',
        help='Do not preserve directory structure in output (flat structure)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Do not include subject/activity/trial columns in CSV'
    )
    parser.add_argument(
        '--combined-csv',
        type=str,
        help='Create a single combined CSV file with all data'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("="*60)
    logger.info("NPZ TO CSV CONVERTER")
    logger.info("="*60)
    logger.info(f"Input directory:    {args.input_dir}")
    logger.info(f"Output directory:   {args.output_dir}")
    logger.info(f"Include metadata:   {not args.no_metadata}")
    logger.info(f"Preserve structure: {not args.no_structure}")

    # Create converter
    converter = NPZToCSVConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        include_metadata=not args.no_metadata,
        preserve_structure=not args.no_structure
    )

    # Convert files
    if args.single_file:
        npz_path = Path(args.single_file)
        if not npz_path.exists():
            logger.error(f"File not found: {npz_path}")
            sys.exit(1)
        converter.convert_file(npz_path)
    elif args.combined_csv:
        npz_files = converter.find_npz_files()
        if npz_files:
            combined_output = Path(args.combined_csv)
            combined_output.parent.mkdir(parents=True, exist_ok=True)
            converter.create_combined_csv(npz_files, combined_output)
    else:
        converter.convert_all()

    logger.info("Conversion completed!")

if __name__ == "__main__":
    main()
