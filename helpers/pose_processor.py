import cv2
import tensorflow as tf
import numpy as np
from .visualization_utils import draw_prediction_on_image_simple, draw_prediction_on_image_adaptive, draw_prediction_on_image_enhanced
from .feedback_utils import PoseFeedback, draw_comprehensive_feedback_overlay

class PoseProcessor:
    """Handles pose detection processing with improved stability and squat form analysis."""
    
    def __init__(self, movenet_model, input_size):
        self.movenet = movenet_model
        self.input_size = input_size
        self.feedback = PoseFeedback()
        
        # Responsive smoothing parameters - optimized for stability without lag
        self.prev_keypoints = None
        self.smoothing_factor = 0.6  # Reduced for more responsiveness
        self.confidence_threshold = 0.15  # Minimum confidence for keypoints
        
        # Keypoint-specific smoothing factors - balanced for responsiveness
        self.keypoint_smoothing_factors = {
            0: 0.7,   # nose - balanced
            1: 0.7,   # left_eye
            2: 0.7,   # right_eye
            3: 0.7,   # left_ear
            4: 0.7,   # right_ear
            5: 0.65,  # left_shoulder - core body part
            6: 0.65,  # right_shoulder
            7: 0.6,   # left_elbow - arm part
            8: 0.6,   # right_elbow
            9: 0.55,  # left_wrist - extremity
            10: 0.55, # right_wrist
            11: 0.65, # left_hip - core body part
            12: 0.65, # right_hip
            13: 0.6,  # left_knee - leg part
            14: 0.6,  # right_knee
            15: 0.55, # left_ankle - extremity
            16: 0.55  # right_ankle
        }
    
    def get_keypoint_confidence(self, keypoints, kp_idx):
        """Get confidence score for a keypoint, handling different formats."""
        try:
            if keypoints.shape[1] == 3:
                # Format: (x, y, confidence)
                return float(keypoints[kp_idx, 2])
            elif keypoints.shape[1] == 1:
                # Format: (confidence) - this is likely the scores array
                return float(keypoints[kp_idx, 0])
            else:
                # Unknown format, assume high confidence
                return 1.0
        except (IndexError, TypeError):
            # If we can't access the confidence, assume it's high
            return 1.0
    
    def get_keypoint_coords(self, keypoints, kp_idx):
        """Get coordinates for a keypoint, handling different formats."""
        try:
            if keypoints.shape[1] == 3:
                # Format: (x, y, confidence)
                return float(keypoints[kp_idx, 0]), float(keypoints[kp_idx, 1])
            elif keypoints.shape[1] == 1:
                # This might be just confidence scores, need to check actual format
                # For now, return dummy coordinates
                return 0.5, 0.5
            else:
                # Unknown format
                return 0.5, 0.5
        except (IndexError, TypeError):
            return 0.5, 0.5
    
    def calculate_movement(self, current_kp, previous_kp):
        """Calculate movement between current and previous keypoints."""
        if previous_kp is None:
            return np.zeros(current_kp.shape[0])
        
        # Calculate Euclidean distance for each keypoint
        movement = np.zeros(current_kp.shape[0])
        for i in range(current_kp.shape[0]):
            try:
                curr_x, curr_y = self.get_keypoint_coords(current_kp, i)
                prev_x, prev_y = self.get_keypoint_coords(previous_kp, i)
                movement[i] = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            except:
                movement[i] = 0.0
        
        return movement
    
    def apply_responsive_smoothing(self, current_keypoints, prev_keypoints):
        """Apply responsive smoothing that reduces jitter without causing lag."""
        if prev_keypoints is None:
            return current_keypoints
        
        smoothed_keypoints = current_keypoints.copy()
        
        # Calculate movement for each keypoint
        movement = self.calculate_movement(current_keypoints, prev_keypoints)
        
        for i in range(current_keypoints.shape[0]):
            try:
                # Get confidence for this keypoint
                confidence = self.get_keypoint_confidence(current_keypoints, i)
                
                # Skip if confidence is too low
                if confidence < self.confidence_threshold:
                    continue
                
                # Get keypoint-specific smoothing factor
                base_smoothing = self.keypoint_smoothing_factors.get(i, 0.6)
                
                # Adaptive smoothing based on movement
                if movement[i] < 0.01:  # Very small movement - apply more smoothing
                    smoothing_factor = min(0.8, base_smoothing + 0.1)
                elif movement[i] < 0.05:  # Small movement - moderate smoothing
                    smoothing_factor = base_smoothing
                else:  # Large movement - minimal smoothing for responsiveness
                    smoothing_factor = max(0.3, base_smoothing - 0.2)
                
                # Apply smoothing - handle different keypoint formats
                if current_keypoints.shape[1] == 3 and prev_keypoints.shape[1] == 3:
                    # Standard format: (x, y, confidence)
                    smoothed_keypoints[i] = (smoothing_factor * prev_keypoints[i] + 
                                           (1 - smoothing_factor) * current_keypoints[i])
                elif current_keypoints.shape[1] == 1 and prev_keypoints.shape[1] == 1:
                    # Confidence scores only
                    smoothed_keypoints[i, 0] = (smoothing_factor * prev_keypoints[i, 0] + 
                                              (1 - smoothing_factor) * current_keypoints[i, 0])
            except Exception as e:
                # If smoothing fails for this keypoint, keep original
                continue
        
        return smoothed_keypoints
    
    def apply_smoothing(self, current_keypoints, prev_keypoints):
        """Apply responsive smoothing pipeline."""
        if prev_keypoints is None:
            return current_keypoints
        
        try:
            # Apply responsive smoothing
            smoothed_keypoints = self.apply_responsive_smoothing(current_keypoints, prev_keypoints)
            return smoothed_keypoints
        except Exception as e:
            # If smoothing fails, return original keypoints
            print(f"Warning: Smoothing failed, using original keypoints. Error: {e}")
            return current_keypoints
    
    def process_frame(self, frame, show_feedback=True):
        """Process a single frame and return the result."""
        # Convert BGR to RGB for model input
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad to model input size
        input_image = tf.expand_dims(image_rgb, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)

        # Run MoveNet
        keypoints_with_scores = self.movenet(input_image)

        # Apply responsive smoothing
        if self.prev_keypoints is not None:
            keypoints_with_scores = self.apply_smoothing(keypoints_with_scores, self.prev_keypoints)
        
        # Store current keypoints for next frame
        self.prev_keypoints = keypoints_with_scores.copy()

        # Get comprehensive feedback if requested
        feedback = None
        if show_feedback:
            feedback = self.feedback.get_comprehensive_feedback(
                keypoints_with_scores, frame.shape[0], frame.shape[1]
            )

        # Determine threshold and visualization method based on feedback
        if feedback and feedback.get('form_analysis'):
            # Use enhanced visualization for squat analysis
            threshold = 0.15
            use_enhanced = True
        elif feedback and feedback['distance_status'] in ['very_close', 'close']:
            threshold = 0.15
            use_adaptive = True
            use_enhanced = False
        else:
            threshold = 0.2
            use_adaptive = False
            use_enhanced = False

        # Draw prediction with appropriate method
        if use_enhanced:
            output_overlay = draw_prediction_on_image_enhanced(
                frame.copy(),
                keypoints_with_scores,
                keypoint_threshold=threshold
            )
        elif use_adaptive:
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

        # Add comprehensive feedback overlay if requested
        if show_feedback and feedback:
            output_overlay = draw_comprehensive_feedback_overlay(output_overlay, feedback)

        return output_overlay, keypoints_with_scores, feedback

def process_video_with_squat_analysis(video_path, movenet_model, input_size, output_path=None):
    """Process video with comprehensive squat form analysis - optimized for mobile."""
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
    print("Squat Form Analysis System (Mobile Optimized):")
    print("- Analyzes back rounding, knee alignment, depth, and arm position")
    print("- Provides real-time form score and recommendations")
    print("- Detects squat phases: standing, descending, bottom, ascending")
    print("- Enhanced keypoint visualization for better form analysis")
    print("- Compact feedback overlay for better visibility")
    print("- Mobile-friendly resolution for better performance")
    print("- Green: Good form | Orange: Needs improvement | Red: Form issues")
    
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

            # Display the result in a mobile-friendly window
            cv2.namedWindow('MoveNet Lightning - Squat Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('MoveNet Lightning - Squat Analysis', 1280, 720)  # 720p window
            cv2.imshow('MoveNet Lightning - Squat Analysis', output_overlay)
            
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

def process_webcam_with_squat_analysis(movenet_model, input_size):
    """Process webcam feed with comprehensive squat form analysis - optimized for mobile."""
    processor = PoseProcessor(movenet_model, input_size)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set camera properties for mobile-friendly resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 720p width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 720p height
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available

    print("MoveNet Lightning - Squat Form Analysis (Mobile Optimized)")
    print("Form Analysis Features:")
    print("- Back rounding detection")
    print("- Knee alignment and valgus detection")
    print("- Squat depth analysis")
    print("- Arm position feedback")
    print("- Real-time form scoring")
    print("- Enhanced keypoint visualization")
    print("- Compact feedback overlay")
    print("- 720p resolution for mobile performance")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            output_overlay, keypoints_with_scores, feedback = processor.process_frame(frame, show_feedback=True)

            # Display the result in a mobile-friendly window
            cv2.namedWindow('MoveNet Lightning - Squat Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('MoveNet Lightning - Squat Analysis', 1280, 720)  # 720p window
            cv2.imshow('MoveNet Lightning - Squat Analysis', output_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Keep the old functions for backward compatibility
def process_video_with_improved_feedback(video_path, movenet_model, input_size, output_path=None):
    """Process video with improved feedback system (legacy function)."""
    return process_video_with_squat_analysis(video_path, movenet_model, input_size, output_path)

def process_webcam_with_improved_feedback(movenet_model, input_size):
    """Process webcam feed with improved feedback system (legacy function)."""
    return process_webcam_with_squat_analysis(movenet_model, input_size) 