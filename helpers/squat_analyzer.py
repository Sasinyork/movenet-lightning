import cv2
import numpy as np
import math

class SquatFormAnalyzer:
    """Analyzes squat form and provides feedback on common issues."""
    
    def __init__(self):
        self.prev_keypoints = None
        self.squat_phase = "standing"  # standing, descending, bottom, ascending
        self.phase_frames = 0
        self.form_issues = []
        self.feedback_history = []
        
        # Phase detection parameters
        self.hip_positions = []  # Store recent hip positions for movement detection
        self.max_history = 10  # Number of frames to track for movement
        self.movement_threshold = 5  # Minimum pixel movement to detect direction
        self.standing_threshold = -50  # Hip must be this much above knee to be "standing" (negative in image coords)
        self.bottom_threshold = 10  # Hip must be this much below knee to be "bottom" (positive in image coords)
        
        # Keypoint indices for MoveNet
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
        
    def get_keypoint_coords(self, keypoints, index, image_height, image_width):
        """Get pixel coordinates for a keypoint."""
        if keypoints[index, 2] > 0.15:  # Confidence threshold
            x = int(keypoints[index, 1] * image_width)
            y = int(keypoints[index, 0] * image_height)
            return (x, y)
        return None
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (point2 is the vertex)."""
        if point1 is None or point2 is None or point3 is None:
            return None
            
        # Calculate vectors
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def detect_movement_direction(self, current_hip_y):
        """Detect if the person is moving up or down based on hip position history."""
        if len(self.hip_positions) < 3:
            return "unknown"
        
        # Calculate recent movement trend
        recent_positions = self.hip_positions[-5:]  # Last 5 frames
        if len(recent_positions) < 3:
            return "unknown"
        
        # Calculate average movement over recent frames
        total_movement = 0
        for i in range(1, len(recent_positions)):
            movement = recent_positions[i] - recent_positions[i-1]
            total_movement += movement
        
        avg_movement = total_movement / (len(recent_positions) - 1)
        
        # Determine direction based on movement
        if avg_movement > self.movement_threshold:
            return "descending"  # Hip moving down (y increasing)
        elif avg_movement < -self.movement_threshold:
            return "ascending"   # Hip moving up (y decreasing)
        else:
            return "stable"       # Minimal movement
    
    def detect_squat_phase(self, keypoints, image_height, image_width):
        """Detect the current phase of the squat movement with improved logic."""
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        
        if left_hip and right_hip and left_knee and right_knee:
            # Calculate average hip and knee positions
            hip_y = (left_hip[1] + right_hip[1]) / 2
            knee_y = (left_knee[1] + right_knee[1]) / 2
            
            # Calculate relative position (hip should be above knee in standing)
            hip_knee_diff = hip_y - knee_y
            
            # Update hip position history for movement detection
            self.hip_positions.append(hip_y)
            if len(self.hip_positions) > self.max_history:
                self.hip_positions.pop(0)
            
            # Detect movement direction
            movement_direction = self.detect_movement_direction(hip_y)
            
            # Determine phase based on position AND movement
            # Standing: Hip well above knee (negative difference in image coords)
            if hip_knee_diff < self.standing_threshold:
                if movement_direction == "stable":
                    new_phase = "standing"
                elif movement_direction == "descending":
                    new_phase = "descending"
                else:
                    new_phase = "standing"  # Default to standing if unclear
                    
            # Bottom: Hip below knee level (positive difference in image coords)
            elif hip_knee_diff > self.bottom_threshold:
                if movement_direction == "ascending":
                    new_phase = "ascending"
                elif movement_direction == "stable":
                    new_phase = "bottom"
                else:
                    new_phase = "bottom"  # Default to bottom if unclear
                    
            # Transition zone: Between standing and bottom
            else:
                if movement_direction == "descending":
                    new_phase = "descending"
                elif movement_direction == "ascending":
                    new_phase = "ascending"
                elif hip_knee_diff > 0:  # Hip below knee (positive in image coords)
                    new_phase = "bottom"
                else:  # Hip above knee but not high enough for standing
                    new_phase = "descending"  # Still going down
            
            # Update phase tracking
            if new_phase == self.squat_phase:
                self.phase_frames += 1
            else:
                self.squat_phase = new_phase
                self.phase_frames = 0
            
            return new_phase, hip_knee_diff
        
        return "unknown", 0
    
    def analyze_back_form(self, keypoints, image_height, image_width):
        """Analyze back form for rounding issues."""
        issues = []
        
        # Get keypoints for back analysis
        nose = self.get_keypoint_coords(keypoints, self.NOSE, image_height, image_width)
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Calculate shoulder and hip centers
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            
            # Calculate back angle (should be relatively straight)
            if nose:
                # Calculate angle between nose-shoulder-hip
                back_angle = self.calculate_angle(nose, shoulder_center, hip_center)
                
                if back_angle:
                    # Check for excessive forward lean (back rounding)
                    if back_angle < 150:  # Should be closer to 180 degrees for straight back
                        issues.append({
                            'type': 'back_rounding',
                            'severity': 'high' if back_angle < 140 else 'medium',
                            'message': f'Back is rounding (angle: {back_angle:.1f}°)',
                            'recommendation': 'Keep chest up and back straight'
                        })
            
            # Check for lateral tilt (shoulders should be level)
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            if shoulder_diff > 15:  # More than 15 pixels difference
                issues.append({
                    'type': 'shoulder_tilt',
                    'severity': 'medium',
                    'message': 'Shoulders are not level',
                    'recommendation': 'Keep shoulders level and square'
                })
        
        return issues
    
    def analyze_knee_form(self, keypoints, image_height, image_width):
        """Analyze knee position and alignment."""
        issues = []
        
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        left_ankle = self.get_keypoint_coords(keypoints, self.LEFT_ANKLE, image_height, image_width)
        right_ankle = self.get_keypoint_coords(keypoints, self.RIGHT_ANKLE, image_height, image_width)
        
        if left_knee and right_knee and left_ankle and right_ankle:
            # Check knee alignment with toes
            left_knee_toe_alignment = abs(left_knee[0] - left_ankle[0])
            right_knee_toe_alignment = abs(right_knee[0] - right_ankle[0])
            
            # Knees should be roughly aligned with toes (within reasonable range)
            if left_knee_toe_alignment > 30 or right_knee_toe_alignment > 30:
                issues.append({
                    'type': 'knee_alignment',
                    'severity': 'high',
                    'message': 'Knees are not aligned with toes',
                    'recommendation': 'Keep knees in line with toes'
                })
            
            # Check for knee valgus (knees caving in)
            knee_distance = abs(left_knee[0] - right_knee[0])
            ankle_distance = abs(left_ankle[0] - right_ankle[0])
            
            if knee_distance < ankle_distance * 0.8:  # Knees too close together
                issues.append({
                    'type': 'knee_valgus',
                    'severity': 'high',
                    'message': 'Knees are caving inward',
                    'recommendation': 'Push knees outward, keep them aligned'
                })
        
        return issues
    
    def analyze_depth(self, keypoints, image_height, image_width):
        """Analyze squat depth."""
        issues = []
        
        left_hip = self.get_keypoint_coords(keypoints, self.LEFT_HIP, image_height, image_width)
        right_hip = self.get_keypoint_coords(keypoints, self.RIGHT_HIP, image_height, image_width)
        left_knee = self.get_keypoint_coords(keypoints, self.LEFT_KNEE, image_height, image_width)
        right_knee = self.get_keypoint_coords(keypoints, self.RIGHT_KNEE, image_height, image_width)
        
        if left_hip and right_hip and left_knee and right_knee:
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            knee_center = ((left_knee[0] + right_knee[0]) / 2, 
                           (left_knee[1] + right_knee[1]) / 2)
            
            # Calculate depth (hip should go below knee level for proper squat)
            depth_ratio = (hip_center[1] - knee_center[1]) / image_height
            
            if self.squat_phase == "bottom":
                if depth_ratio > -0.1:  # Hip not low enough
                    issues.append({
                        'type': 'insufficient_depth',
                        'severity': 'medium',
                        'message': 'Squat depth is insufficient',
                        'recommendation': 'Go deeper - thighs should be parallel to ground'
                    })
                elif depth_ratio < -0.3:  # Too deep
                    issues.append({
                        'type': 'excessive_depth',
                        'severity': 'low',
                        'message': 'Squat is very deep',
                        'recommendation': 'Consider reducing depth if uncomfortable'
                    })
        
        return issues
    
    def analyze_arm_position(self, keypoints, image_height, image_width):
        """Analyze arm position during squat."""
        issues = []
        
        left_shoulder = self.get_keypoint_coords(keypoints, self.LEFT_SHOULDER, image_height, image_width)
        right_shoulder = self.get_keypoint_coords(keypoints, self.RIGHT_SHOULDER, image_height, image_width)
        left_elbow = self.get_keypoint_coords(keypoints, self.LEFT_ELBOW, image_height, image_width)
        right_elbow = self.get_keypoint_coords(keypoints, self.RIGHT_ELBOW, image_height, image_width)
        left_wrist = self.get_keypoint_coords(keypoints, self.LEFT_WRIST, image_height, image_width)
        right_wrist = self.get_keypoint_coords(keypoints, self.RIGHT_WRIST, image_height, image_width)
        
        if left_shoulder and right_shoulder and left_elbow and right_elbow:
            # Check if arms are raised (common in bodyweight squats)
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            elbow_center = ((left_elbow[0] + right_elbow[0]) / 2, 
                           (left_elbow[1] + right_elbow[1]) / 2)
            
            # Arms should be roughly at shoulder level or extended forward
            arm_angle = self.calculate_angle(shoulder_center, elbow_center, 
                                           (elbow_center[0], elbow_center[1] - 50))
            
            if arm_angle and arm_angle < 45:  # Arms too low
                issues.append({
                    'type': 'arm_position',
                    'severity': 'low',
                    'message': 'Arms could be raised higher',
                    'recommendation': 'Extend arms forward or raise them higher'
                })
        
        return issues
    
    def analyze_squat_form(self, keypoints_with_scores, image_height, image_width):
        """Comprehensive squat form analysis."""
        keypoints = keypoints_with_scores[0, 0, :, :]
        
        # Detect squat phase
        phase, depth_metric = self.detect_squat_phase(keypoints, image_height, image_width)
        
        # Analyze different aspects of form
        back_issues = self.analyze_back_form(keypoints, image_height, image_width)
        knee_issues = self.analyze_knee_form(keypoints, image_height, image_width)
        depth_issues = self.analyze_depth(keypoints, image_height, image_width)
        arm_issues = self.analyze_arm_position(keypoints, image_height, image_width)
        
        # Combine all issues
        all_issues = back_issues + knee_issues + depth_issues + arm_issues
        
        # Create comprehensive feedback
        feedback = {
            'phase': phase,
            'phase_frames': self.phase_frames,
            'issues': all_issues,
            'depth_metric': depth_metric,
            'overall_score': self.calculate_form_score(all_issues),
            'primary_issue': self.get_primary_issue(all_issues),
            'recommendations': self.get_recommendations(all_issues)
        }
        
        # Update feedback history
        self.feedback_history.append(feedback)
        if len(self.feedback_history) > 10:  # Keep last 10 frames
            self.feedback_history.pop(0)
        
        return feedback
    
    def calculate_form_score(self, issues):
        """Calculate overall form score (0-100)."""
        if not issues:
            return 100
        
        # Weight issues by severity
        severity_weights = {'low': 5, 'medium': 15, 'high': 25}
        total_penalty = sum(severity_weights.get(issue['severity'], 10) for issue in issues)
        
        return max(0, 100 - total_penalty)
    
    def get_primary_issue(self, issues):
        """Get the most important issue to address."""
        if not issues:
            return None
        
        # Prioritize by severity and type
        priority_order = ['back_rounding', 'knee_valgus', 'knee_alignment', 
                         'insufficient_depth', 'shoulder_tilt', 'arm_position']
        
        for priority_type in priority_order:
            for issue in issues:
                if issue['type'] == priority_type:
                    return issue
        
        return issues[0]  # Return first issue if no priority match
    
    def get_recommendations(self, issues):
        """Get actionable recommendations."""
        if not issues:
            return ["Form looks good! Keep it up!"]
        
        recommendations = []
        for issue in issues:
            if issue['recommendation'] not in recommendations:
                recommendations.append(issue['recommendation'])
        
        return recommendations[:3]  # Limit to top 3 recommendations 