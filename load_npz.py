import numpy as np
import argparse
import os

def load_and_display_npz(npz_file_path, show_frames=5, show_joints=5):
    """Load NPZ and actually display the pose data"""
    
    # Check file size
    size_mb = os.path.getsize(npz_file_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    
    # Load data
    data = np.load(npz_file_path)
    
    try:
        if 'reconstruction' not in data:
            print("No 'reconstruction' key found")
            return
        
        reconstruction = data['reconstruction']
        print(f"Data shape: {reconstruction.shape}")
        print(f"Data type: {reconstruction.dtype}")
        
        if len(reconstruction.shape) == 4:  # (num_people, frames, joints, coords)
            num_people, frames, joints, coords = reconstruction.shape
            print(f"\nFormat: {num_people} people, {frames} frames, {joints} joints, {coords} coordinates")
            
            for person_idx in range(num_people):
                print(f"\n{'='*60}")
                print(f"PERSON {person_idx + 1} DATA:")
                print(f"{'='*60}")
                
                person_data = reconstruction[person_idx]
                
                # Show first few frames
                frames_to_show = min(show_frames, frames)
                for frame_idx in range(180):
                    print(f"\nFrame {frame_idx + 1}:")
                    print("-" * 40)
                    
                    frame_data = person_data[frame_idx]
                    joints_to_show = min(show_joints, joints)
                    
                    print("Joint#  X-coord   Y-coord   Z-coord")
                    print("-" * 40)
                    for joint_idx in range(joints_to_show):
                        x, y, z = frame_data[joint_idx]
                        print(f"{joint_idx:4d}:  {x:8.3f}  {y:8.3f}  {z:8.3f}")
                    
                    if joints > show_joints:
                        print(f"... and {joints - show_joints} more joints")
                
                if frames > show_frames:
                    print(f"\n... and {frames - show_frames} more frames")
        
        elif len(reconstruction.shape) == 3:  # (frames, joints, coords) - single person
            frames, joints, coords = reconstruction.shape
            print(f"\nFormat: Single person, {frames} frames, {joints} joints, {coords} coordinates")
            
            print(f"\n{'='*60}")
            print(f"POSE DATA:")
            print(f"{'='*60}")
            
            # Show first few frames
            frames_to_show = min(show_frames, frames)
            for frame_idx in range(frames_to_show):
                print(f"\nFrame {frame_idx + 1}:")
                print("-" * 40)
                
                frame_data = reconstruction[frame_idx]
                joints_to_show = min(show_joints, joints)
                
                print("Joint#  X-coord   Y-coord   Z-coord")
                print("-" * 40)
                for joint_idx in range(joints_to_show):
                    x, y, z = frame_data[joint_idx]
                    print(f"{joint_idx:4d}:  {x:8.3f}  {y:8.3f}  {z:8.3f}")
                
                if joints > show_joints:
                    print(f"... and {joints - show_joints} more joints")
            
            if frames > show_frames:
                print(f"\n... and {frames - show_frames} more frames")
        
        else:
            print(f"Unexpected data shape: {reconstruction.shape}")
            print("Raw data preview:")
            print(reconstruction)
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS:")
        print(f"{'='*60}")
        print(f"X coordinate range: {reconstruction[:,:,:,0].min():.3f} to {reconstruction[:,:,:,0].max():.3f}")
        print(f"Y coordinate range: {reconstruction[:,:,:,1].min():.3f} to {reconstruction[:,:,:,1].max():.3f}")
        print(f"Z coordinate range: {reconstruction[:,:,:,2].min():.3f} to {reconstruction[:,:,:,2].max():.3f}")
        
    finally:
        data.close()

def show_specific_frame(npz_file_path, frame_idx, person_idx=0):
    """Show all data for a specific frame"""
    data = np.load(npz_file_path)
    
    try:
        reconstruction = data['reconstruction']
        
        if len(reconstruction.shape) == 4:  # Multiple people
            if person_idx >= reconstruction.shape[0]:
                print(f"Person {person_idx} not found. Available: 0-{reconstruction.shape[0]-1}")
                return
            frame_data = reconstruction[person_idx, frame_idx]
        else:  # Single person
            frame_data = reconstruction[frame_idx]
        
        print(f"Frame {frame_idx + 1}, Person {person_idx + 1}:")
        print("=" * 50)
        print("Joint#  X-coordinate  Y-coordinate  Z-coordinate")
        print("-" * 50)
        
        for joint_idx, (x, y, z) in enumerate(frame_data):
            print(f"{joint_idx:4d}:    {x:10.4f}    {y:10.4f}    {z:10.4f}")
    
    finally:
        data.close()

def main():
    parser = argparse.ArgumentParser(description='Display NPZ pose data')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to NPZ file')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to show (default: 5)')
    parser.add_argument('--joints', type=int, default=17, help='Number of joints to show per frame (default: 17)')
    parser.add_argument('--frame', type=int, help='Show specific frame (shows all joints)')
    parser.add_argument('--person', type=int, default=0, help='Person index for specific frame (default: 0)')
    parser.add_argument('--all-frames', action='store_true', help='Show all frames (warning: lots of output)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return
    
    if args.frame is not None:
        show_specific_frame(args.file, args.frame, args.person)
    else:
        frames_to_show = 999999 if args.all_frames else args.frames
        load_and_display_npz(args.file, frames_to_show, args.joints)

if __name__ == "__main__":
    main()
