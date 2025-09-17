#!/usr/bin/env python3
"""
Batch Process Videos for Skeleton Generation
Automatically processes all videos in the GMDC5A24 dataset and generates skeleton NPZ files
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import datetime
import json
import traceback

# Import the skeleton generation function from gen_skes
from gen_skes import generate_skeletons

# Configure logging
def setup_logging(log_file: str = None):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)

class VideoProcessor:
    """Class to handle batch video processing for skeleton generation"""
    
    def __init__(self, input_root: str, output_root: str, num_person: int = 1):
        """
        Initialize the video processor
        
        Args:
            input_root: Root directory containing video dataset
            output_root: Root directory for output files
            num_person: Number of persons to detect (1 or 2)
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.num_person = num_person
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = {
            'total_videos': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'processing_times': []
        }
        
        # Ensure output directory exists
        self.output_root.mkdir(parents=True, exist_ok=True)
        
    def find_videos(self) -> List[Dict[str, Path]]:
        """
        Find all video files in the dataset
        
        Returns:
            List of dictionaries containing video paths and metadata
        """
        videos = []
        dataset_path = self.input_root / 'ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76'
        
        if not dataset_path.exists():
            self.logger.error(f"Dataset path not found: {dataset_path}")
            return videos
        
        # Iterate through subjects
        for subject_dir in sorted(dataset_path.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith('Subject'):
                continue
                
            subject_id = subject_dir.name
            
            # Process both ADL and Fall directories
            for activity_type in ['ADL', 'Fall']:
                activity_dir = subject_dir / activity_type
                if not activity_dir.exists():
                    continue
                    
                # Find all MP4 files
                for video_file in sorted(activity_dir.glob('*.mp4')):
                    videos.append({
                        'path': video_file,
                        'subject': subject_id,
                        'activity': activity_type,
                        'filename': video_file.name,
                        'relative_path': video_file.relative_to(dataset_path)
                    })
        
        self.stats['total_videos'] = len(videos)
        self.logger.info(f"Found {len(videos)} videos to process")
        return videos
    
    def get_output_path(self, video_info: Dict) -> Path:
        """
        Generate output path maintaining directory structure
        
        Args:
            video_info: Dictionary containing video metadata
            
        Returns:
            Output path for NPZ file
        """
        # Create output path: output/GMDC5A24/Subject_X/Activity/filename.npz
        output_path = self.output_root / 'GMDC5A24' / video_info['subject'] / video_info['activity']
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Replace .mp4 with .npz
        npz_filename = video_info['filename'].replace('.mp4', '.npz')
        return output_path / npz_filename
    
    def process_video(self, video_info: Dict, force: bool = False) -> bool:
        """
        Process a single video to generate skeleton
        
        Args:
            video_info: Dictionary containing video metadata
            force: Force reprocessing even if output exists
            
        Returns:
            Success status
        """
        video_path = video_info['path']
        output_path = self.get_output_path(video_info)
        
        # Check if already processed
        if output_path.exists() and not force:
            self.logger.info(f"Skipping (already exists): {video_info['relative_path']}")
            self.stats['skipped'] += 1
            return True
        
        self.logger.info(f"Processing: {video_info['relative_path']}")
        start_time = time.time()
        
        try:
            # Call the generate_skeletons function
            generate_skeletons(
                video=str(video_path),
                output_animation=False,
                num_person=self.num_person,
                ab_dis=False
            )
            
            # Move the generated NPZ file to the correct location
            # The original function saves to './output/[video_name].npz'
            temp_output = Path('./output') / video_info['filename'].replace('.mp4', '.npz')
            
            if temp_output.exists():
                # Ensure parent directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Move file to correct location
                temp_output.rename(output_path)
                
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)
                self.stats['processed'] += 1
                
                self.logger.info(f"✓ Completed in {processing_time:.2f}s: {output_path}")
                return True
            else:
                raise FileNotFoundError(f"Expected output file not found: {temp_output}")
                
        except Exception as e:
            self.logger.error(f"✗ Failed to process {video_path}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            self.stats['failed'] += 1
            return False
    
    def process_batch(self, videos: List[Dict], force: bool = False, 
                     start_idx: int = 0, end_idx: int = None):
        """
        Process a batch of videos
        
        Args:
            videos: List of video information dictionaries
            force: Force reprocessing
            start_idx: Starting index for batch processing
            end_idx: Ending index for batch processing
        """
        if end_idx is None:
            end_idx = len(videos)
        
        videos_batch = videos[start_idx:end_idx]
        self.logger.info(f"Processing batch: {start_idx} to {end_idx} ({len(videos_batch)} videos)")
        
        for i, video_info in enumerate(videos_batch, start=start_idx):
            self.logger.info(f"[{i+1}/{len(videos)}] Processing video...")
            self.process_video(video_info, force=force)
    
    def save_statistics(self):
        """Save processing statistics to file"""
        stats_file = self.output_root / 'processing_stats.json'
        
        # Calculate average processing time
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            self.stats['avg_processing_time'] = avg_time
        
        self.stats['timestamp'] = datetime.now().isoformat()
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        self.logger.info(f"Statistics saved to {stats_file}")
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total videos found: {self.stats['total_videos']}")
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"Skipped (existing): {self.stats['skipped']}")
        print(f"Failed: {self.stats['failed']}")
        
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            print(f"Average processing time: {avg_time:.2f} seconds")
            print(f"Total processing time: {sum(self.stats['processing_times']):.2f} seconds")
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Batch process videos for skeleton generation'
    )
    parser.add_argument(
        '--input-root', 
        type=str, 
        default='./input_videos',
        help='Root directory containing video dataset'
    )
    parser.add_argument(
        '--output-root', 
        type=str, 
        default='./output',
        help='Root directory for output files'
    )
    parser.add_argument(
        '--num-person', 
        type=int, 
        default=1,
        choices=[1, 2],
        help='Number of persons to detect (1 or 2)'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force reprocess existing files'
    )
    parser.add_argument(
        '--start-idx', 
        type=int, 
        default=0,
        help='Start index for batch processing'
    )
    parser.add_argument(
        '--end-idx', 
        type=int, 
        default=None,
        help='End index for batch processing'
    )
    parser.add_argument(
        '--log-file', 
        type=str, 
        default='processing.log',
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting batch video processing")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create processor
    processor = VideoProcessor(
        input_root=args.input_root,
        output_root=args.output_root,
        num_person=args.num_person
    )
    
    # Find videos
    videos = processor.find_videos()
    
    if not videos:
        logger.error("No videos found to process")
        return
    
    # Process videos
    try:
        processor.process_batch(
            videos=videos,
            force=args.force,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    finally:
        # Save statistics
        processor.save_statistics()

if __name__ == "__main__":
    main()
