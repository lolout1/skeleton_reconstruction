#!/usr/bin/env python3
"""
Batch Process Videos for Skeleton Generation
Automatically processes all videos in the GMDC5A24 dataset and generates skeleton NPZ files
with robust filename conversion support.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import json
import traceback
import shutil

# Import the skeleton generation function and filename converter from gen_skes
from gen_skes import generate_skeletons, FilenameConverter

# Configure logging
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing logging config
    )
    return logging.getLogger(__name__)

class VideoProcessor:
    """Class to handle batch video processing for skeleton generation with filename conversion"""

    def __init__(self, input_root: str, output_root: str, num_person: int = 1, 
                 rf: int = 27, use_standardized_names: bool = True):
        """
        Initialize the video processor

        Args:
            input_root: Root directory containing video dataset
            output_root: Root directory for output files
            num_person: Number of persons to detect (1 or 2)
            rf: Receptive field size (27 or 81)
            use_standardized_names: Whether to use standardized naming convention
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.num_person = num_person
        self.rf = rf
        self.use_standardized_names = use_standardized_names
        self.logger = logging.getLogger(__name__)
        
        # Initialize filename converter
        self.filename_converter = FilenameConverter()

        # Statistics tracking
        self.stats = {
            'total_videos': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'processing_times': [],
            'errors': []
        }

        # Ensure output directory exists
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Video processor initialized:")
        self.logger.info(f"  Input root: {self.input_root}")
        self.logger.info(f"  Output root: {self.output_root}")
        self.logger.info(f"  Persons to detect: {self.num_person}")
        self.logger.info(f"  Receptive field: {self.rf}")
        self.logger.info(f"  Standardized names: {self.use_standardized_names}")

    def find_videos(self) -> List[Dict[str, any]]:
        """
        Find all video files in the dataset

        Returns:
            List of dictionaries containing video paths and metadata
        """
        videos = []
        
        # Look for the dataset directory
        dataset_paths = [
            self.input_root / 'ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76',
            self.input_root  # In case videos are directly in input_root
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if path.exists():
                dataset_path = path
                break
        
        if not dataset_path:
            self.logger.error(f"Dataset not found in any of these paths: {dataset_paths}")
            return videos

        self.logger.info(f"Using dataset path: {dataset_path}")

        # Iterate through subjects
        for subject_dir in sorted(dataset_path.iterdir()):
            if not subject_dir.is_dir():
                continue
                
            # Handle both "Subject X" and "Subject_X" formats
            subject_name = subject_dir.name
            if not (subject_name.startswith('Subject') or 'subject' in subject_name.lower()):
                continue

            # Process both ADL and Fall directories
            for activity_type in ['ADL', 'Fall']:
                activity_dir = subject_dir / activity_type
                if not activity_dir.exists():
                    # Try lowercase
                    activity_dir = subject_dir / activity_type.lower()
                    if not activity_dir.exists():
                        continue

                # Find all video files (multiple extensions)
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
                for ext_pattern in video_extensions:
                    for video_file in sorted(activity_dir.glob(ext_pattern)):
                        videos.append({
                            'path': video_file,
                            'subject': subject_name,
                            'activity': activity_type,
                            'filename': video_file.name,
                            'relative_path': video_file.relative_to(self.input_root),
                            'absolute_path': str(video_file)
                        })

        self.stats['total_videos'] = len(videos)
        self.logger.info(f"Found {len(videos)} videos to process")
        
        # Log some examples
        if videos:
            self.logger.info("Sample videos found:")
            for i, video in enumerate(videos[:3]):
                self.logger.info(f"  {i+1}. {video['relative_path']}")
            if len(videos) > 3:
                self.logger.info(f"  ... and {len(videos)-3} more")
        
        return videos

    def get_output_paths(self, video_info: Dict[str, any]) -> Tuple[Path, Path]:
        """
        Generate both old and new format output paths

        Args:
            video_info: Dictionary containing video metadata

        Returns:
            Tuple of (structured_output_path, standardized_output_path)
        """
        # Create structured output path: output/GMDC5A24/Subject_X/Activity/filename.npz
        structured_path = (self.output_root / 'GMDC5A24' / 
                          video_info['subject'] / video_info['activity'])
        structured_path.mkdir(parents=True, exist_ok=True)
        structured_npz = structured_path / video_info['filename'].replace('.mp4', '.npz')
        
        # Create standardized output path using filename converter
        if self.use_standardized_names:
            standardized_filename = self.filename_converter.convert(
                video_info['absolute_path'], extension='.npz'
            )
            standardized_npz = self.output_root / standardized_filename
        else:
            standardized_npz = structured_npz
        
        return structured_npz, standardized_npz

    def get_expected_gen_skes_output(self, video_info: Dict[str, any]) -> Path:
        """
        Get the filename that gen_skes.py will actually create

        Args:
            video_info: Dictionary containing video metadata

        Returns:
            Path where gen_skes.py will save the output
        """
        if self.use_standardized_names:
            # gen_skes.py now uses FilenameConverter, so get the standardized name
            standardized_filename = self.filename_converter.convert(
                video_info['absolute_path'], extension='.npz'
            )
            return self.output_root / standardized_filename
        else:
            # Fallback to original naming
            return self.output_root / video_info['filename'].replace('.mp4', '.npz')

    def process_video(self, video_info: Dict[str, any], force: bool = False) -> bool:
        """
        Process a single video to generate skeleton

        Args:
            video_info: Dictionary containing video metadata
            force: Force reprocessing even if output exists

        Returns:
            Success status
        """
        video_path = video_info['path']
        structured_output, standardized_output = self.get_output_paths(video_info)
        expected_gen_output = self.get_expected_gen_skes_output(video_info)

        # Check if already processed (check both possible output locations)
        existing_outputs = [structured_output, standardized_output, expected_gen_output]
        existing_file = None
        
        for output_path in existing_outputs:
            if output_path.exists():
                existing_file = output_path
                break

        if existing_file and not force:
            self.logger.info(f"✓ Skipping (already exists): {existing_file.name}")
            self.stats['skipped'] += 1
            return True

        self.logger.info(f"Processing: {video_info['relative_path']}")
        self.logger.info(f"Expected output: {expected_gen_output.name}")
        
        start_time = time.time()

        try:
            # Validate video file exists
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Call the generate_skeletons function
            self.logger.debug(f"Calling generate_skeletons with:")
            self.logger.debug(f"  video: {video_path}")
            self.logger.debug(f"  rf: {self.rf}")
            self.logger.debug(f"  num_person: {self.num_person}")

            generate_skeletons(
                video=str(video_path),
                rf=self.rf,
                output_animation=False,
                num_person=self.num_person,
                ab_dis=False
            )

            # Check if the expected output was created
            if expected_gen_output.exists():
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)
                self.stats['processed'] += 1

                # If we want structured organization, copy to structured location
                if self.use_standardized_names and expected_gen_output != structured_output:
                    try:
                        structured_output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(expected_gen_output, structured_output)
                        self.logger.debug(f"Copied to structured location: {structured_output}")
                    except Exception as e:
                        self.logger.warning(f"Failed to copy to structured location: {e}")

                self.logger.info(f"✓ Completed in {processing_time:.2f}s: {expected_gen_output.name}")
                return True
            else:
                # Check if output exists in any alternative location
                alt_outputs = [
                    self.output_root / video_info['filename'].replace('.mp4', '.npz'),
                    Path('./output') / video_info['filename'].replace('.mp4', '.npz')
                ]
                
                found_output = None
                for alt_path in alt_outputs:
                    if alt_path.exists():
                        found_output = alt_path
                        break
                
                if found_output:
                    # Move to expected location
                    found_output.rename(expected_gen_output)
                    processing_time = time.time() - start_time
                    self.stats['processing_times'].append(processing_time)
                    self.stats['processed'] += 1
                    self.logger.info(f"✓ Moved and completed in {processing_time:.2f}s: {expected_gen_output.name}")
                    return True
                else:
                    raise FileNotFoundError(f"No output file found after processing. Expected: {expected_gen_output}")

        except Exception as e:
            error_msg = f"Failed to process {video_info['relative_path']}: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            self.logger.debug(traceback.format_exc())
            self.stats['failed'] += 1
            self.stats['errors'].append({
                'video': str(video_info['relative_path']),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    def process_batch(self, videos: List[Dict[str, any]], force: bool = False,
                     start_idx: int = 0, end_idx: Optional[int] = None) -> None:
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
        self.logger.info(f"Processing batch: indices {start_idx}-{end_idx-1} ({len(videos_batch)} videos)")

        batch_start_time = time.time()
        
        for i, video_info in enumerate(videos_batch, start=start_idx):
            try:
                self.logger.info(f"[{i+1}/{len(videos)}] Processing video...")
                success = self.process_video(video_info, force=force)
                
                # Log progress every 10 videos
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - batch_start_time
                    avg_time = elapsed / (i + 1 - start_idx) if i > start_idx else 0
                    remaining = (len(videos_batch) - (i + 1 - start_idx)) * avg_time
                    self.logger.info(f"Progress: {i+1}/{len(videos)} videos, "
                                   f"ETA: {remaining/60:.1f} minutes")
                    
            except KeyboardInterrupt:
                self.logger.warning("Processing interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error processing video {i+1}: {e}")
                continue

    def save_statistics(self) -> None:
        """Save processing statistics to file"""
        stats_file = self.output_root / 'processing_stats.json'

        # Calculate average processing time
        if self.stats['processing_times']:
            self.stats['avg_processing_time'] = (
                sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            )
            self.stats['total_processing_time'] = sum(self.stats['processing_times'])

        self.stats['timestamp'] = datetime.now().isoformat()
        self.stats['configuration'] = {
            'num_person': self.num_person,
            'rf': self.rf,
            'use_standardized_names': self.use_standardized_names,
            'input_root': str(self.input_root),
            'output_root': str(self.output_root)
        }

        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            self.logger.info(f"Statistics saved to {stats_file}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")

        self.print_summary()

    def print_summary(self) -> None:
        """Print processing summary"""
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"Total videos found:      {self.stats['total_videos']}")
        print(f"Successfully processed:  {self.stats['processed']}")
        print(f"Skipped (existing):      {self.stats['skipped']}")
        print(f"Failed:                  {self.stats['failed']}")
        
        success_rate = (self.stats['processed'] / max(1, self.stats['total_videos'])) * 100
        print(f"Success rate:            {success_rate:.1f}%")

        if self.stats['processing_times']:
            avg_time = self.stats['avg_processing_time']
            total_time = self.stats['total_processing_time']
            print(f"Average processing time: {avg_time:.2f} seconds")
            print(f"Total processing time:   {total_time/60:.1f} minutes")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for i, error in enumerate(self.stats['errors'][:5], 1):
                print(f"  {i}. {error['video']}: {error['error'][:50]}...")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors'])-5} more errors")
        
        print("="*70)

def main():
    """Main function with enhanced argument parsing"""
    parser = argparse.ArgumentParser(
        description='Batch process videos for skeleton generation with filename conversion',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help='Number of persons to detect'
    )
    parser.add_argument(
        '--rf', '--receptive-field',
        type=int,
        default=27,
        choices=[27, 81],
        help='Receptive field size for temporal modeling'
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
    parser.add_argument(
        '--no-standardized-names',
        action='store_true',
        help='Disable standardized naming convention'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("STARTING BATCH VIDEO PROCESSING")
    logger.info("="*60)
    logger.info(f"Configuration: {vars(args)}")

    # Validate input directory
    if not os.path.exists(args.input_root):
        logger.error(f"Input directory does not exist: {args.input_root}")
        sys.exit(1)

    # Create processor
    processor = VideoProcessor(
        input_root=args.input_root,
        output_root=args.output_root,
        num_person=args.num_person,
        rf=args.rf,
        use_standardized_names=not args.no_standardized_names
    )

    # Find videos
    videos = processor.find_videos()

    if not videos:
        logger.error("No videos found to process")
        sys.exit(1)

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
    except Exception as e:
        logger.error(f"Unexpected error during batch processing: {e}")
        logger.debug(traceback.format_exc())
    finally:
        # Save statistics
        processor.save_statistics()

    logger.info("Batch processing completed")

if __name__ == "__main__":
    main()
