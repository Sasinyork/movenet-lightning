# main.py
import tensorflow as tf
import numpy as np
import cv2
import os
import helpers.visualization_utils as vis
from helpers.model_utils import load_movenet_model

# Global variables for model
movenet = None
input_size = None

# Initialize variables for smoothing
prev_keypoints = None
smoothing_factor = 0.7  # Adjust this value (0-1) for smoothing

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

def init_crop_region(image_height, image_width):
    """Defines the default crop region."""
    if image_width > image_height:
        box_height = image_width / 9
        box_width = image_height / 9
        y_min = (image_height / 2) - (box_height / 2)
        x_min = (image_width / 2) - (box_width / 2)
    else:
        box_height = image_width / 9
        box_width = image_height / 9
        y_min = (image_height / 2) - (box_height / 2)
        x_min = (image_width / 2) - (box_width / 2)
    
    x_max = x_min + box_width
    y_max = y_min + box_height
    
    return {
        'y_min': y_min / image_height,
        'x_min': x_min / image_width,
        'y_max': y_max / image_height,
        'x_max': x_max / image_width,
        'height': box_height / image_height,
        'width': box_width / image_width
    }

def determine_crop_region(keypoints_with_scores, image_height, image_width, region):
    """Determines the region to crop the image for the model to run inference on."""
    keypoints = keypoints_with_scores[0, 0, :, :]
    
    # Find the center of the detected person
    valid_keypoints = keypoints[keypoints[:, 2] > 0.1]
    if len(valid_keypoints) > 0:
        center_y = np.mean(valid_keypoints[:, 0])
        center_x = np.mean(valid_keypoints[:, 1])
        
        # Calculate bounding box
        y_min = np.min(valid_keypoints[:, 0]) - 0.1
        y_max = np.max(valid_keypoints[:, 0]) + 0.1
        x_min = np.min(valid_keypoints[:, 1]) - 0.1
        x_max = np.max(valid_keypoints[:, 1]) + 0.1
        
        # Ensure bounds are within image
        y_min = max(0, y_min)
        y_max = min(1, y_max)
        x_min = max(0, x_min)
        x_max = min(1, x_max)
        
        # Update region
        region['y_min'] = y_min
        region['x_min'] = x_min
        region['y_max'] = y_max
        region['x_max'] = x_max
        region['height'] = y_max - y_min
        region['width'] = x_max - x_min
    
    return region

def apply_smoothing(current_keypoints, prev_keypoints, smoothing_factor):
    """Apply temporal smoothing to keypoints."""
    if prev_keypoints is None:
        return current_keypoints
    
    smoothed_keypoints = smoothing_factor * prev_keypoints + (1 - smoothing_factor) * current_keypoints
    return smoothed_keypoints

def process_video(video_path, output_path=None):
    """Process a video file and save the output with pose detection."""
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
    
    # Initialize crop region
    crop_region = None
    prev_keypoints = None
    frame_count = 0
    
    print("Processing video... Press 'q' to stop early.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Initialize crop region on first frame
            if crop_region is None:
                crop_region = init_crop_region(height, width)

            # Convert BGR to RGB for model input
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop the image to focus on the person
            y_min = int(crop_region['y_min'] * height)
            y_max = int(crop_region['y_max'] * height)
            x_min = int(crop_region['x_min'] * width)
            x_max = int(crop_region['x_max'] * width)
            
            # Ensure crop region is valid
            y_min = max(0, y_min)
            y_max = min(height, y_max)
            x_min = max(0, x_min)
            x_max = min(width, x_max)
            
            # Crop the image
            cropped_image = image_rgb[y_min:y_max, x_min:x_max]
            
            if cropped_image.size == 0:
                # If crop region is invalid, use full image
                cropped_image = image_rgb
            
            # Resize and pad to model input size
            input_image = tf.expand_dims(cropped_image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

            # Run MoveNet
            keypoints_with_scores = movenet(input_image)

            # Apply smoothing
            if prev_keypoints is not None:
                keypoints_with_scores = apply_smoothing(keypoints_with_scores, prev_keypoints, smoothing_factor)
            prev_keypoints = keypoints_with_scores.copy()

            # Update crop region based on detected keypoints
            crop_region = determine_crop_region(keypoints_with_scores, height, width, crop_region)

            # Draw prediction using the simple OpenCV-based function
            output_overlay = vis.draw_prediction_on_image_simple(
                frame.copy(),  # Use BGR frame directly
                keypoints_with_scores,
                keypoint_threshold=0.2  # Increased threshold for better quality
            )

            # Draw crop region rectangle
            cv2.rectangle(output_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display the result
            cv2.imshow('MoveNet Lightning - Video Processing', output_overlay)
            
            # Save frame if writer is available
            if writer:
                writer.write(output_overlay)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing stopped by user.")
                break
                
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Video processing completed!")

def process_webcam():
    """Process webcam feed in real-time."""
    # Open the default webcam (0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize crop region
    crop_region = None
    prev_keypoints = None

    print("Press 'q' to quit.")
    print("Press 'r' to reset crop region.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Initialize crop region on first frame
            if crop_region is None:
                height, width = frame.shape[:2]
                crop_region = init_crop_region(height, width)

            # Convert BGR to RGB for model input
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop the image to focus on the person
            height, width = image_rgb.shape[:2]
            y_min = int(crop_region['y_min'] * height)
            y_max = int(crop_region['y_max'] * height)
            x_min = int(crop_region['x_min'] * width)
            x_max = int(crop_region['x_max'] * width)
            
            # Ensure crop region is valid
            y_min = max(0, y_min)
            y_max = min(height, y_max)
            x_min = max(0, x_min)
            x_max = min(width, x_max)
            
            # Crop the image
            cropped_image = image_rgb[y_min:y_max, x_min:x_max]
            
            if cropped_image.size == 0:
                # If crop region is invalid, use full image
                cropped_image = image_rgb
            
            # Resize and pad to model input size
            input_image = tf.expand_dims(cropped_image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

            # Run MoveNet
            keypoints_with_scores = movenet(input_image)

            # Apply smoothing
            if prev_keypoints is not None:
                keypoints_with_scores = apply_smoothing(keypoints_with_scores, prev_keypoints, smoothing_factor)
            prev_keypoints = keypoints_with_scores.copy()

            # Update crop region based on detected keypoints
            crop_region = determine_crop_region(keypoints_with_scores, height, width, crop_region)

            # Draw prediction using the simple OpenCV-based function
            output_overlay = vis.draw_prediction_on_image_simple(
                frame.copy(),  # Use BGR frame directly
                keypoints_with_scores,
                keypoint_threshold=0.2  # Increased threshold for better quality
            )

            # Draw crop region rectangle
            cv2.rectangle(output_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display the result
            cv2.imshow('MoveNet Lightning - Webcam', output_overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset crop region
                crop_region = None
                prev_keypoints = None
                print("Crop region reset")

    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to choose between webcam and video processing."""
    print("MoveNet Lightning Pose Detection")
    print("=" * 40)
    print("Available Models:")
    print("1. Lightning (faster, 192x192 input)")
    print("2. Thunder (more accurate, 256x256 input)")
    print("3. Lightning TFLite (optimized)")
    print("4. Thunder TFLite (optimized)")
    print("=" * 40)
    
    # Model selection
    while True:
        model_choice = input("Select model (1-4): ").strip()
        if model_choice == "1":
            model_name = "movenet_lightning"
            break
        elif model_choice == "2":
            model_name = "movenet_thunder"
            break
        elif model_choice == "3":
            model_name = "movenet_lightning_f16"
            break
        elif model_choice == "4":
            model_name = "movenet_thunder_f16"
            break
        else:
            print("Invalid choice. Please enter 1-4.")
    
    print(f"\nLoading {model_name}...")
    global movenet, input_size
    movenet, input_size = load_movenet_model(model_name)
    print(f"Model loaded! Input size: {input_size}x{input_size}")
    
    print("\n" + "=" * 40)
    print("1. Use Webcam (real-time)")
    print("2. Process Video File")
    print("=" * 40)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nStarting webcam...")
            process_webcam()
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
            
            process_video(video_path, output_path)
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
