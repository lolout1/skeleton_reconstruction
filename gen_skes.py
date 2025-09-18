#!/usr/bin/env python3
"""
GAST-Net 3D Pose Estimation - Enhanced Generation Script
Generates 3D human pose skeletons from video with robust filename conversion.
"""

import torch
import sys
import os
import os.path as osp
import argparse
import cv2
import time
import h5py
import re
import logging
from tqdm import tqdm
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project paths
sys.path.insert(0, osp.dirname(osp.realpath(__file__)))
from tools.utils import get_path
from model.gast_net import SpatioTemporalModel, SpatioTemporalModelOptimized1f
from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton
from common.generators import *
from tools.vis_h36m import render_animation
from tools.preprocess import load_kpts_json, h36m_coco_format, revise_kpts, revise_skes
from tools.inference import gen_pose
from tools.vis_kpts import plot_keypoint

# Get project paths
cur_dir, chk_root, data_root, lib_root, output_root = get_path(__file__)
model_dir = chk_root + 'gastnet/'

# Add lib path for pose estimation
sys.path.insert(1, lib_root)
from lib.pose import gen_video_kpts as hrnet_pose
sys.path.pop(1)
sys.path.pop(0)

# Skeleton configuration
skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
)
adj = adj_mx_from_skeleton(skeleton)

# Joint mappings
joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
keypoints_metadata = {
    'keypoints_symmetry': (joints_left, joints_right), 
    'layout_name': 'Human3.6M', 
    'num_joints': 17
}
width, height = (1920, 1080)


class FilenameConverter:
    """Robust filename converter for video paths to standardized naming format."""
    
    def __init__(self):
        self.subject_patterns = [
            r'Subject[\s_]*(\d+)',  # Subject 1, Subject_1
            r'S(\d+)',              # S1, S01
            r'subject[\s_]*(\d+)',  # subject 1 (lowercase)
            r'P(\d+)',              # P1 (Person)
        ]
        
        self.activity_mapping = {
            'ADL': 'A01',
            'adl': 'A01', 
            'Activities of Daily Living': 'A01',
            'daily': 'A01',
            'normal': 'A01',
            'Fall': 'A10',
            'fall': 'A10',
            'falls': 'A10',
            'falling': 'A10',
        }
        
        self.video_patterns = [
            r'(\d+)\.mp4$',         # 06.mp4
            r'(\d+)\.avi$',         # 06.avi
            r'[Tt](\d+)',           # T06, t06
            r'[Vv]ideo[\s_]*(\d+)', # Video 6, video_6
            r'(\d+)$',              # Just number at end
        ]
    
    def extract_subject_number(self, path):
        """Extract subject number from path using multiple patterns."""
        for pattern in self.subject_patterns:
            match = re.search(pattern, path, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Fallback: look for any number in path segments
        path_parts = Path(path).parts
        for part in path_parts:
            numbers = re.findall(r'\d+', part)
            if numbers:
                return int(numbers[0])
        
        logger.warning(f"Could not extract subject number from {path}, using default 1")
        return 1
    
    def extract_activity_code(self, path):
        """Extract activity code from path."""
        path_lower = path.lower()
        
        # Direct mapping check
        for activity_name, code in self.activity_mapping.items():
            if activity_name.lower() in path_lower:
                return code
        
        # Check path segments
        path_parts = Path(path).parts
        for part in path_parts:
            part_lower = part.lower()
            for activity_name, code in self.activity_mapping.items():
                if activity_name.lower() in part_lower:
                    return code
        
        # Default fallback based on common keywords
        if any(keyword in path_lower for keyword in ['fall', 'emergency']):
            return 'A10'
        else:
            logger.warning(f"Could not determine activity type from {path}, using ADL (A01)")
            return 'A01'
    
    def extract_video_number(self, path):
        """Extract video/trial number from path."""
        filename = Path(path).name
        
        # Try patterns in order of specificity
        for pattern in self.video_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Fallback: extract any number from filename
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])  # Take the last number found
        
        logger.warning(f"Could not extract video number from {path}, using default 1")
        return 1
    
    def convert(self, video_path, extension='.npz'):
        """
        Convert video path to standardized format.
        
        Args:
            video_path (str): Input video path
            extension (str): Output file extension
            
        Returns:
            str: Standardized filename (e.g., 'S01_A01_T06.npz')
        """
        try:
            # Normalize path separators
            normalized_path = str(Path(video_path))
            
            # Extract components
            subject_num = self.extract_subject_number(normalized_path)
            activity_code = self.extract_activity_code(normalized_path)
            video_num = self.extract_video_number(normalized_path)
            
            # Format components
            subject_formatted = f"S{subject_num:02d}"
            video_formatted = f"T{video_num:02d}"
            
            # Create standardized filename
            standardized_name = f"{subject_formatted}_{activity_code}_{video_formatted}{extension}"
            
            logger.info(f"Converted '{Path(video_path).name}' → '{standardized_name}'")
            return standardized_name
            
        except Exception as e:
            logger.error(f"Error converting filename {video_path}: {e}")
            # Fallback to original filename
            fallback_name = Path(video_path).stem + extension
            logger.warning(f"Using fallback filename: {fallback_name}")
            return fallback_name


def load_model_realtime(rf=81):
    """Load GAST-Net model for real-time inference."""
    model_configs = {
        27: {'checkpoint': '27_frame_model_causal.bin', 'filters': [3, 3, 3], 'channels': 128},
        81: {'checkpoint': '81_frame_model_causal.bin', 'filters': [3, 3, 3, 3], 'channels': 64}
    }
    
    if rf not in model_configs:
        raise ValueError(f'Only support {list(model_configs.keys())} receptive field models for inference!')
    
    config = model_configs[rf]
    chk = model_dir + config['checkpoint']
    
    logger.info('Loading GAST-Net for real-time inference...')
    model_pos = SpatioTemporalModelOptimized1f(
        adj, 17, 2, 17, 
        filter_widths=config['filters'], 
        causal=True,
        channels=config['channels'], 
        dropout=0.25
    )
    
    # Load pre-trained model
    checkpoint = torch.load(chk, map_location='cpu')
    model_pos.load_state_dict(checkpoint['model_pos'])
    
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        logger.info('Model loaded on GPU')
    else:
        logger.info('Model loaded on CPU')
    
    model_pos.eval()
    logger.info('GAST-Net successfully loaded')
    return model_pos


def load_model_layer(rf=27):
    """Load GAST-Net model for layer-wise inference."""
    model_configs = {
        27: {'checkpoint': '27_frame_model.bin', 'filters': [3, 3, 3], 'channels': 128},
        81: {'checkpoint': '81_frame_model.bin', 'filters': [3, 3, 3, 3], 'channels': 64}
    }
    
    if rf not in model_configs:
        raise ValueError(f'Only support {list(model_configs.keys())} receptive field models for inference!')
    
    config = model_configs[rf]
    chk = model_dir + config['checkpoint']
    
    if not os.path.exists(chk):
        raise FileNotFoundError(f"Model checkpoint not found: {chk}")
    
    logger.info('Loading GAST-Net for layer-wise inference...')
    model_pos = SpatioTemporalModel(
        adj, 17, 2, 17, 
        filter_widths=config['filters'], 
        channels=config['channels'], 
        dropout=0.05
    )
    
    # Load pre-trained model
    checkpoint = torch.load(chk, map_location='cpu')
    model_pos.load_state_dict(checkpoint['model_pos'])
    
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        logger.info('Model loaded on GPU')
    else:
        logger.info('Model loaded on CPU')
    
    model_pos.eval()
    logger.info('GAST-Net successfully loaded')
    return model_pos


def validate_video_file(video_path):
    """Validate video file exists and is readable."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Test if OpenCV can open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if frame_count == 0:
        raise ValueError(f"Video file appears to be empty: {video_path}")
    
    logger.info(f"Video validated: {frame_count} frames, {fps:.2f} FPS")
    return True


def generate_skeletons(video='', rf=27, output_animation=False, num_person=1, ab_dis=False, custom_output_name=None):
    """
    Generate 3D skeletons from video with enhanced filename conversion.
    
    Args:
        video (str): Input video path
        rf (int): Receptive field size (27 or 81)
        output_animation (bool): Generate animation video
        num_person (int): Number of people to track (1 or 2)
        ab_dis (bool): Whether to generate absolute distance
        custom_output_name (str): Custom output filename (optional)
    """
    
    # Initialize filename converter
    filename_converter = FilenameConverter()
    
    # Validate inputs
    if not video:
        raise ValueError("Video path cannot be empty")
    
    validate_video_file(video)
    
    if num_person not in [1, 2]:
        raise ValueError("num_person must be 1 or 2")
    
    if rf not in [27, 81]:
        raise ValueError("rf (receptive field) must be 27 or 81")
    
    # Ensure output directory exists
    os.makedirs(output_root, exist_ok=True)
    
    logger.info(f"Processing video: {video}")
    logger.info(f"Parameters: rf={rf}, num_person={num_person}, animation={output_animation}")
    
    # Read video properties
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    
    try:
        # Generate 2D keypoints
        logger.info("Generating 2D pose keypoints...")
        keypoints, scores = hrnet_pose(video, det_dim=416, num_peroson=num_person, gen_output=True)
        
        if keypoints is None or scores is None:
            raise RuntimeError("Failed to generate 2D keypoints")
        
        # Process keypoints
        logger.info("Processing keypoints...")
        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        re_kpts = revise_kpts(keypoints, scores, valid_frames)
        num_person = len(re_kpts)
        
        if num_person == 0:
            raise RuntimeError("No valid keypoints detected")
        
        logger.info(f"Detected {num_person} person(s)")
        
        # Load 3D pose model
        logger.info("Loading 3D pose estimation model...")
        model_pos = load_model_layer(rf)
        
        # Generate 3D poses
        logger.info("Generating 3D human poses...")
        pad = (rf - 1) // 2
        causal_shift = 0
        
        prediction = gen_pose(re_kpts, valid_frames, width, height, model_pos, pad, causal_shift)
        
        # Post-process predictions
        if num_person == 2:
            prediction = revise_skes(prediction, re_kpts, valid_frames)
        elif ab_dis:
            prediction[0][:, :, 2] -= np.expand_dims(
                np.amin(prediction[0][:, :, 2], axis=1), axis=1
            ).repeat([17], axis=1)
        else:
            prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])
        
        # Generate output filename
        if custom_output_name:
            base_filename = custom_output_name
        else:
            base_filename = filename_converter.convert(video, extension='')
        
        # Save results
        if output_animation:
            # Generate animation
            animation_filename = f"animation_{base_filename}.mp4"
            viz_output = os.path.join(output_root, animation_filename)
            
            logger.info("Generating animation...")
            
            # Prepare data for animation
            anim_output = {}
            for i, anim_prediction in enumerate(prediction):
                anim_output.update({f'Reconstruction {i+1}': anim_prediction})
            
            # Set coordinate system flag
            same_coord = (num_person == 2)
            
            # Prepare keypoints for rendering
            re_kpts_transposed = re_kpts.transpose(1, 0, 2, 3)  # (M, T, N, 2) -> (T, M, N, 2)
            
            render_animation(
                re_kpts_transposed, keypoints_metadata, anim_output, skeleton, 
                25, 30000, np.array(70., dtype=np.float32),
                viz_output, input_video_path=video, viewport=(width, height), 
                com_reconstrcution=same_coord
            )
            
            logger.info(f"Animation saved: {viz_output}")
            
        else:
            # Save 3D coordinates
            npz_filename = f"{base_filename}.npz"
            output_npz = os.path.join(output_root, npz_filename)
            
            logger.info("Saving 3D reconstruction...")
            np.savez_compressed(output_npz, reconstruction=prediction)
            logger.info(f"3D reconstruction saved: {output_npz}")
        
        logger.info("Processing completed successfully!")
        return prediction
        
    except Exception as e:
        logger.error(f"Error during skeleton generation: {e}")
        raise


def parse_arguments():
    """Parse command line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description='GAST-Net 3D Pose Estimation - Generate skeletons from video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-v', '--video', 
        type=str, 
        default='baseball.mp4',
        help='Input video filename (relative to data/video/ or absolute path)'
    )
    
    parser.add_argument(
        '-rf', '--receptive-field', 
        type=int, 
        default=81, 
        choices=[27, 81],
        help='Receptive field size for temporal modeling'
    )
    
    parser.add_argument(
        '-a', '--animation', 
        action='store_true',
        help='Generate animation video output'
    )
    
    parser.add_argument(
        '-np', '--num-person', 
        type=int, 
        default=1, 
        choices=[1, 2],
        help='Number of people to track and estimate'
    )
    
    parser.add_argument(
        '--ab-dis', 
        action='store_true',
        help='Generate absolute distance for 3D poses'
    )
    
    parser.add_argument(
        '-o', '--output-name',
        type=str,
        help='Custom output filename (without extension)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Determine video path
    if os.path.isabs(args.video):
        video_path = args.video
    else:
        video_path = os.path.join(data_root, 'video', args.video)
    
    try:
        # Generate skeletons
        prediction = generate_skeletons(
            video=video_path,
            rf=args.receptive_field,
            output_animation=args.animation,
            num_person=args.num_person,
            ab_dis=args.ab_dis,
            custom_output_name=args.output_name
        )
        
        logger.info("All processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
