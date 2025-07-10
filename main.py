# main.py
import tensorflow as tf
import numpy as np
import cv2
import os
import helpers.visualization_utils as vis
from helpers.model_utils import load_movenet_model
from helpers.pose_processor import process_video_with_improved_feedback, process_webcam_with_improved_feedback

# Global variables for model
movenet = None
input_size = None

def get_unique_output_path(base_path, suffix="_pose_detected"):
    """Generate a unique output path by incrementing the filename if it already exists."""
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the original filename
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    # Start with the base name
    counter = 0
    while True:
        if counter == 0:
            # First try without number
            output_filename = f"{name}{suffix}{ext}"
        else:
            # Then try with incrementing numbers
            output_filename = f"{name}{suffix}_{counter}{ext}"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # If file doesn't exist, we can use this path
        if not os.path.exists(output_path):
            return output_path
        
        counter += 1

def main():
    """Main function to choose between webcam and video processing."""
    print("MoveNet Lightning Pose Detection (Mobile Optimized)")
    print("=" * 50)
    print("Using MoveNet Lightning TFLite model (192x192 input)")
    print("Optimized for mobile/Android development")
    print("=" * 50)
    
    print("\nLoading MoveNet Lightning TFLite model...")
    global movenet, input_size
    movenet, input_size = load_movenet_model()
    print(f"Model loaded! Input size: {input_size}x{input_size}")
    
    print("\n" + "=" * 40)
    print("1. Use Webcam (real-time)")
    print("2. Process Video File")
    print("3. Use Webcam with Improved Feedback")
    print("4. Process Video with Improved Feedback")
    print("=" * 40)
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\nStarting webcam...")
            # Simple webcam without feedback
            process_webcam_simple()
            break
        elif choice == "2":
            print("\nVideo file processing selected.")
            video_path = input("Enter the path to your video file: ").strip()
            
            if not os.path.exists(video_path):
                print(f"Error: File '{video_path}' not found!")
                continue
            
            # Ask if user wants to save output
            save_output = input("Save processed video? (y/n): ").strip().lower()
            output_path = None
            if save_output == 'y':
                # Create output filename
                output_path = get_unique_output_path(video_path)
                print(f"Output will be saved as: {output_path}")
            
            # Simple video processing without feedback
            process_video_simple(video_path, output_path)
            break
        elif choice == "3":
            print("\nStarting webcam with improved feedback...")
            process_webcam_with_improved_feedback(movenet, input_size)
            break
        elif choice == "4":
            print("\nVideo file processing with improved feedback selected.")
            video_path = input("Enter the path to your video file: ").strip()
            
            if not os.path.exists(video_path):
                print(f"Error: File '{video_path}' not found!")
                continue
            
            # Ask if user wants to save output
            save_output = input("Save processed video? (y/n): ").strip().lower()
            output_path = None
            if save_output == 'y':
                # Create output filename
                output_path = get_unique_output_path(video_path, "_with_improved_feedback")
                print(f"Output will be saved as: {output_path}")
            
            process_video_with_improved_feedback(video_path, movenet, input_size, output_path)
            break
        else:
            print("Invalid choice. Please enter 1-4.")

# Keep the simple processing functions for backward compatibility
def process_video_simple(video_path, output_path=None):
    """Simple video processing without feedback."""
    from helpers.pose_processor import PoseProcessor
    
    processor = PoseProcessor(movenet, input_size)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    
    print("Processing video... Press 'q' to stop early.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Process frame without feedback
            output_overlay, keypoints_with_scores, feedback = processor.process_frame(frame, show_feedback=False)

            # Display the result
            cv2.imshow('MoveNet Lightning - Video Processing', output_overlay)
            
            if writer:
                writer.write(output_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing stopped by user.")
                break
                
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Video processing completed!")

def process_webcam_simple():
    """Simple webcam processing without feedback."""
    from helpers.pose_processor import PoseProcessor
    
    processor = PoseProcessor(movenet, input_size)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame without feedback
            output_overlay, keypoints_with_scores, feedback = processor.process_frame(frame, show_feedback=False)

            # Display the result
            cv2.imshow('MoveNet Lightning - Webcam', output_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
