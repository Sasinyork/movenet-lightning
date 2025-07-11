import cv2
import numpy as np
from .squat_analyzer import SquatFormAnalyzer

class PoseFeedback:
    """Handles real-time feedback for pose detection positioning and squat form."""
    
    def __init__(self):
        self.stable_frames = 0
        self.last_person_center = None
        self.last_person_size = None
        self.feedback_history = []
        self.last_distance_feedback = None
        self.frames_since_movement = 0
        self.movement_threshold = 0.05  # Reduced sensitivity
        self.stability_frames_needed = 15  # Need to hold still for 15 frames (0.5 seconds at 30fps)
        self.feedback_persistence_frames = 90  # Keep feedback for 3 seconds (90 frames at 30fps)
        
        # Initialize squat form analyzer
        self.squat_analyzer = SquatFormAnalyzer()
        self.form_analysis_enabled = True
    
    def calculate_person_center_and_size(self, keypoints_with_scores, image_height, image_width):
        """Calculate the center and size of the person more reliably."""
        keypoints = keypoints_with_scores[0, 0, :, :]
        
        # Use only core body keypoints for more stable detection
        core_keypoints = [5, 6, 11, 12]  # Left/right shoulders and hips
        valid_core_points = []
        
        for i in core_keypoints:
            if keypoints[i, 2] > 0.2:  # Higher threshold for core points
                x = keypoints[i, 1] * image_width
                y = keypoints[i, 0] * image_height
                valid_core_points.append((x, y))
        
        if len(valid_core_points) < 2:
            return None, None, 0, 0
        
        # Calculate center from core body points
        x_coords = [p[0] for p in valid_core_points]
        y_coords = [p[1] for p in valid_core_points]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Calculate size based on core body spread
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Add some padding to account for full body
        estimated_width = width * 1.5
        estimated_height = height * 2.5  # Account for head and legs
        
        return (center_x, center_y), (estimated_width, estimated_height), len(valid_core_points), len(core_keypoints)
    
    def detect_distance_movement(self, current_center, current_size):
        """Detect if the person is moving closer or farther (not just any movement)."""
        if self.last_person_center is None or self.last_person_size is None:
            self.last_person_center = current_center
            self.last_person_size = current_size
            return False, "initializing"
        
        # Calculate change in center position (sideways movement)
        center_change = np.sqrt(
            (current_center[0] - self.last_person_center[0])**2 + 
            (current_center[1] - self.last_person_center[1])**2
        )
        
        # Calculate change in size (indicates distance change)
        size_change = abs(current_size[0] - self.last_person_size[0]) / self.last_person_size[0]
        
        # Update last values
        self.last_person_center = current_center
        self.last_person_size = current_size
        
        # Only consider it movement if there's significant size change (distance change)
        # Ignore small movements and arm movements
        if size_change > 0.1:  # 10% change in size indicates distance movement
            self.frames_since_movement = 0
            return True, "distance_changing"
        elif center_change > self.movement_threshold:
            # Sideways movement, don't reset distance feedback
            return False, "sideways_movement"
        else:
            # No significant movement
            self.frames_since_movement += 1
            return False, "stable"
    
    def get_comprehensive_feedback(self, keypoints_with_scores, image_height, image_width):
        """Get comprehensive feedback including positioning and squat form."""
        keypoints = keypoints_with_scores[0, 0, :, :]
        
        # Calculate person center and size
        center, size, valid_core, total_core = self.calculate_person_center_and_size(
            keypoints_with_scores, image_height, image_width
        )
        
        if center is None or valid_core < 2:
            return {
                'message': "Person not clearly detected",
                'color': (128, 128, 128),
                'distance_status': 'not_detected',
                'recommendation': 'Move into frame and face camera',
                'person_percentage': 0,
                'visibility_percentage': 0,
                'is_stable': False,
                'form_analysis': None
            }
        
        # Calculate person area percentage
        person_area = size[0] * size[1]
        image_area = image_height * image_width
        person_percentage = (person_area / image_area) * 100
        
        # Count visible keypoints
        visible_keypoints = sum(1 for i in range(17) if keypoints[i, 2] > 0.15)
        visibility_percentage = (visible_keypoints / 17) * 100
        
        # Detect movement
        is_moving, movement_type = self.detect_distance_movement(center, size)
        
        # Get squat form analysis if enabled
        form_analysis = None
        if self.form_analysis_enabled and visibility_percentage > 70:  # Only analyze if good visibility
            form_analysis = self.squat_analyzer.analyze_squat_form(keypoints_with_scores, image_height, image_width)
        
        feedback = {
            'message': '',
            'color': (0, 255, 0),
            'distance_status': 'optimal',
            'recommendation': '',
            'person_percentage': person_percentage,
            'visibility_percentage': visibility_percentage,
            'is_stable': False,
            'form_analysis': form_analysis
        }
        
        # Handle different movement scenarios
        if movement_type == "initializing":
            feedback['message'] = "Initializing... hold still"
            feedback['color'] = (0, 165, 255)  # Orange
            feedback['distance_status'] = 'initializing'
            feedback['recommendation'] = 'Stay in place for 2-3 seconds'
            return feedback
        
        elif movement_type == "distance_changing":
            feedback['message'] = "Distance changing... hold still"
            feedback['color'] = (0, 165, 255)  # Orange
            feedback['distance_status'] = 'moving'
            feedback['recommendation'] = 'Stop moving for distance feedback'
            self.last_distance_feedback = None  # Reset previous feedback
            return feedback
        
        elif movement_type == "sideways_movement":
            # Keep previous feedback if it exists and is recent
            if self.last_distance_feedback and self.frames_since_movement < self.feedback_persistence_frames:
                return self.last_distance_feedback
            else:
                feedback['message'] = "Hold still for distance feedback"
                feedback['color'] = (0, 165, 255)  # Orange
                feedback['distance_status'] = 'moving'
                feedback['recommendation'] = 'Stay in place for 2-3 seconds'
                return feedback
        
        elif movement_type == "stable":
            # Check if we've been stable long enough
            if self.frames_since_movement >= self.stability_frames_needed:
                # Generate new distance feedback
                feedback = self._generate_distance_feedback(person_percentage, visibility_percentage, form_analysis)
                feedback['is_stable'] = True
                self.last_distance_feedback = feedback
                return feedback
            else:
                # Still stabilizing, keep previous feedback if available
                if self.last_distance_feedback and self.frames_since_movement < self.feedback_persistence_frames:
                    return self.last_distance_feedback
                else:
                    feedback['message'] = f"Hold still... ({self.stability_frames_needed - self.frames_since_movement} frames left)"
                    feedback['color'] = (0, 165, 255)  # Orange
                    feedback['distance_status'] = 'stabilizing'
                    feedback['recommendation'] = 'Stay in place for distance feedback'
                    return feedback
    
    def _generate_distance_feedback(self, person_percentage, visibility_percentage, form_analysis=None):
        """Generate distance feedback based on person percentage and form analysis."""
        feedback = {
            'message': '',
            'color': (0, 255, 0),
            'distance_status': 'optimal',
            'recommendation': '',
            'person_percentage': person_percentage,
            'visibility_percentage': visibility_percentage,
            'is_stable': True,
            'form_analysis': form_analysis
        }
        
        # Distance thresholds (you can adjust these)
        if person_percentage > 60:
            feedback['message'] = "Too close - step back"
            feedback['color'] = (0, 0, 255)  # Red
            feedback['distance_status'] = 'too_close'
            feedback['recommendation'] = 'Move back 2-3 feet'
        elif person_percentage > 40:
            feedback['message'] = "Good distance for detail"
            feedback['color'] = (0, 165, 255)  # Orange
            feedback['distance_status'] = 'close'
            feedback['recommendation'] = 'Current position is good'
        elif person_percentage > 15:  # Adjusted from 20 to 15
            feedback['message'] = "Perfect distance!"
            feedback['color'] = (0, 255, 0)  # Green
            feedback['distance_status'] = 'optimal'
            feedback['recommendation'] = 'Ideal positioning'
        elif person_percentage > 8:
            feedback['message'] = "Move closer"
            feedback['color'] = (0, 165, 255)  # Orange
            feedback['distance_status'] = 'medium'
            feedback['recommendation'] = 'Step forward 1-2 feet'
        else:
            feedback['message'] = "Too far - move closer"
            feedback['color'] = (0, 0, 255)  # Red
            feedback['distance_status'] = 'too_far'
            feedback['recommendation'] = 'Step forward 3-4 feet'
        
        # Add visibility feedback
        if visibility_percentage < 60:
            feedback['message'] += " - Poor detection"
            feedback['color'] = (0, 0, 255)  # Red
        elif visibility_percentage < 80:
            feedback['message'] += " - Some keypoints missing"
            feedback['color'] = (0, 165, 255)  # Orange
        
        return feedback

def draw_comprehensive_feedback_overlay(image, feedback):
    """Draw comprehensive feedback on the image with compact layout."""
    height, width, _ = image.shape
    
    # Create compact background for text at the bottom
    text_bg_height = 120  # Reduced from 200 to 120
    bg_y_start = height - text_bg_height
    cv2.rectangle(image, (0, bg_y_start), (width, height), (0, 0, 0), -1)
    cv2.rectangle(image, (0, bg_y_start), (width, height), (255, 255, 255), 1)
    
    # Draw main feedback message
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Reduced from 0.6
    thickness = 2
    
    # Main message
    text_size = cv2.getTextSize(feedback['message'], font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = bg_y_start + 25  # Reduced from 35
    cv2.putText(image, feedback['message'], (text_x, text_y), font, font_scale, feedback['color'], thickness)
    
    # Additional info (compact)
    info_text = f"Person: {feedback['person_percentage']:.1f}% | Keypoints: {feedback['visibility_percentage']:.1f}%"
    info_size = cv2.getTextSize(info_text, font, 0.4, 1)[0]  # Reduced from 0.5
    info_x = (width - info_size[0]) // 2
    info_y = bg_y_start + 45  # Reduced from 60
    cv2.putText(image, info_text, (info_x, info_y), font, 0.4, (255, 255, 255), 1)
    
    # Stability indicator (compact)
    if feedback.get('is_stable', False):
        stability_text = "Position Stable ✓"
        stability_color = (0, 255, 0)
    elif feedback['distance_status'] == 'stabilizing':
        stability_text = f"Stabilizing... ({feedback.get('frames_left', 0)})"
        stability_color = (0, 165, 255)
    else:
        stability_text = "Hold Still"
        stability_color = (0, 165, 255)
    
    stability_size = cv2.getTextSize(stability_text, font, 0.35, 1)[0]  # Reduced from 0.4
    stability_x = (width - stability_size[0]) // 2
    stability_y = bg_y_start + 65  # Reduced from 85
    cv2.putText(image, stability_text, (stability_x, stability_y), font, 0.35, stability_color, 1)
    
    # Draw form analysis if available (compact)
    if feedback.get('form_analysis'):
        form_analysis = feedback['form_analysis']
        
        # Form score (compact)
        score = form_analysis.get('overall_score', 0)
        score_color = (0, 255, 0) if score >= 80 else (0, 165, 255) if score >= 60 else (0, 0, 255)
        score_text = f"Form: {score}/100"
        score_size = cv2.getTextSize(score_text, font, 0.4, 1)[0]  # Reduced from 0.5
        score_x = (width - score_size[0]) // 2
        score_y = bg_y_start + 85  # Reduced from 110
        cv2.putText(image, score_text, (score_x, score_y), font, 0.4, score_color, 1)
        
        # Squat phase (compact)
        phase = form_analysis.get('phase', 'unknown')
        phase_text = f"Phase: {phase.title()}"
        phase_size = cv2.getTextSize(phase_text, font, 0.35, 1)[0]  # Reduced from 0.4
        phase_x = (width - phase_size[0]) // 2
        phase_y = bg_y_start + 105  # Reduced from 130
        cv2.putText(image, phase_text, (phase_x, phase_y), font, 0.35, (255, 255, 255), 1)
        
        # Primary issue (compact, only if there's an issue)
        primary_issue = form_analysis.get('primary_issue')
        if primary_issue:
            issue_text = f"Issue: {primary_issue['message']}"
            # Truncate if too long
            if len(issue_text) > 45:  # Reduced from 50
                issue_text = issue_text[:42] + "..."
            issue_size = cv2.getTextSize(issue_text, font, 0.3, 1)[0]  # Reduced from 0.4
            issue_x = (width - issue_size[0]) // 2
            issue_y = bg_y_start + 120  # Reduced from 150
            cv2.putText(image, issue_text, (issue_x, issue_y), font, 0.3, (255, 255, 255), 1)
    
    # Draw recommendation in top corner (smaller)
    if feedback['recommendation']:
        rec_font_scale = 0.35  # Reduced from 0.4
        rec_thickness = 1
        rec_size = cv2.getTextSize(feedback['recommendation'], font, rec_font_scale, rec_thickness)[0]
        rec_x = 10
        rec_y = 25  # Reduced from 30
        cv2.putText(image, feedback['recommendation'], (rec_x, rec_y), font, rec_font_scale, feedback['color'], rec_thickness)
    
    return image

# Keep the old function for backward compatibility
def draw_feedback_overlay(image, feedback):
    """Draw positioning feedback on the image (legacy function)."""
    return draw_comprehensive_feedback_overlay(image, feedback) 