import cv2
import tensorflow as tf
import numpy as np
from .visualization_utils import draw_prediction_on_image_simple, draw_prediction_on_image_adaptive
from .feedback_utils import PoseFeedback, draw_feedback_overlay

class PoseProcessor:
    """Handles pose detection processing with improved stability."""
    
    def __init__(self, movenet_model, input_size):
        self.movenet = movenet_model
        self.input_size = input_size
        self.feedback = PoseFeedback()
        self.prev_keypoints = None
        self.smoothing_factor = 0.7
    
    def apply_smoothing(self, current_keypoints, prev_keypoints):
        """Apply temporal smoothing to keypoints."""
        if prev_keypoints is None:
            return current_keypoints
        
        smoothed_keypoints = self.smoothing_factor * prev_keypoints + (1 - self.smoothing_factor) * current_keypoints
        return smoothed_keypoints
    
    def process_frame(self, frame, show_feedback=True):
        """Process a single frame and return the result."""
        # Convert BGR to RGB for model input
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad to model input size
        input_image = tf.expand_dims(image_rgb, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)

        # Run MoveNet
        keypoints_with_scores = self.movenet(input_image)

        # Apply smoothing
        if self.prev_keypoints is not None:
            keypoints_with_scores = self.apply_smoothing(keypoints_with_scores, self.prev_keypoints)
        self.prev_keypoints = keypoints_with_scores.copy()

        # Get feedback if requested
        feedback = None
        if show_feedback:
            feedback = self.feedback.get_distance_feedback(
                keypoints_with_scores, frame.shape[0], frame.shape[1]
            )

        # Determine threshold based on feedback
        if feedback and feedback['distance_status'] in ['very_close', 'close']:
            threshold = 0.15
            use_adaptive = True
        else:
            threshold = 0.2
            use_adaptive = False

        # Draw prediction
        if use_adaptive:
            output_overlay = draw_prediction_on_image_adaptive(
                frame.copy(),
                keypoints_with_scores,
                keypoint_threshold=threshold
            )
        else:
            output_overlay = draw_prediction_on_image_simple(
                frame.copy(),
                keypoints_with_scores,
                keypoint_threshold=threshold
            )

        # Add feedback overlay if requested
        if show_feedback and feedback:
            output_overlay = draw_feedback_overlay(output_overlay, feedback)

        return output_overlay, keypoints_with_scores, feedback

def process_video_with_improved_feedback(video_path, movenet_model, input_size, output_path=None):
    """Process video with improved feedback system."""
    processor = PoseProcessor(movenet_model, input_size)
    
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
    print("Improved Feedback System:")
    print("- Only gives distance feedback when you hold still")
    print("- Uses core body points for more stable detection")
    print("- Green: Perfect distance")
    print("- Orange: Good, but could be better")
    print("- Red: Too close or too far")
    
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
            
            # Process frame
            output_overlay, keypoints_with_scores, feedback = processor.process_frame(frame, show_feedback=True)

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

def process_webcam_with_improved_feedback(movenet_model, input_size):
    """Process webcam feed with improved feedback system."""
    processor = PoseProcessor(movenet_model, input_size)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("MoveNet Lightning - Webcam with Improved Feedback")
    print("Positioning Guide:")
    print("- Hold still for 2-3 seconds to get distance feedback")
    print("- Green: Perfect distance")
    print("- Orange: Good, but could be better") 
    print("- Red: Too close or too far")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            output_overlay, keypoints_with_scores, feedback = processor.process_frame(frame, show_feedback=True)

            # Display the result
            cv2.imshow('MoveNet Lightning - Webcam', output_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows() 