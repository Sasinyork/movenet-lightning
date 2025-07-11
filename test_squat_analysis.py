#!/usr/bin/env python3
"""
Test script for squat form analysis system.
This script tests the key components of the squat analysis system.
"""

import sys
import os
import numpy as np
import cv2

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.squat_analyzer import SquatFormAnalyzer
from helpers.feedback_utils import PoseFeedback

def test_squat_analyzer():
    """Test the squat analyzer with sample keypoints."""
    print("Testing Squat Form Analyzer...")
    
    analyzer = SquatFormAnalyzer()
    
    # Create sample keypoints for a standing position
    # Format: [y, x, confidence] for each keypoint
    sample_keypoints = np.zeros((1, 1, 17, 3))
    
    # Set up a basic standing pose
    # Head
    sample_keypoints[0, 0, 0, :] = [0.1, 0.5, 0.9]  # Nose
    sample_keypoints[0, 0, 1, :] = [0.08, 0.48, 0.8]  # Left eye
    sample_keypoints[0, 0, 2, :] = [0.08, 0.52, 0.8]  # Right eye
    
    # Shoulders
    sample_keypoints[0, 0, 5, :] = [0.2, 0.4, 0.9]  # Left shoulder
    sample_keypoints[0, 0, 6, :] = [0.2, 0.6, 0.9]  # Right shoulder
    
    # Arms
    sample_keypoints[0, 0, 7, :] = [0.3, 0.35, 0.8]  # Left elbow
    sample_keypoints[0, 0, 8, :] = [0.3, 0.65, 0.8]  # Right elbow
    sample_keypoints[0, 0, 9, :] = [0.4, 0.3, 0.7]   # Left wrist
    sample_keypoints[0, 0, 10, :] = [0.4, 0.7, 0.7]  # Right wrist
    
    # Hips
    sample_keypoints[0, 0, 11, :] = [0.5, 0.4, 0.9]  # Left hip
    sample_keypoints[0, 0, 12, :] = [0.5, 0.6, 0.9]  # Right hip
    
    # Knees
    sample_keypoints[0, 0, 13, :] = [0.7, 0.4, 0.9]  # Left knee
    sample_keypoints[0, 0, 14, :] = [0.7, 0.6, 0.9]  # Right knee
    
    # Ankles
    sample_keypoints[0, 0, 15, :] = [0.9, 0.4, 0.8]  # Left ankle
    sample_keypoints[0, 0, 16, :] = [0.9, 0.6, 0.8]  # Right ankle
    
    # Test the analyzer
    result = analyzer.analyze_squat_form(sample_keypoints, 480, 640)
    
    print(f"Phase: {result['phase']}")
    print(f"Form Score: {result['overall_score']}/100")
    print(f"Number of issues: {len(result['issues'])}")
    
    if result['primary_issue']:
        print(f"Primary issue: {result['primary_issue']['message']}")
    
    print("Recommendations:")
    for rec in result['recommendations']:
        print(f"- {rec}")
    
    print("✓ Squat analyzer test completed successfully!")
    return True

def test_feedback_system():
    """Test the feedback system."""
    print("\nTesting Feedback System...")
    
    feedback = PoseFeedback()
    
    # Create sample keypoints
    sample_keypoints = np.zeros((1, 1, 17, 3))
    
    # Set up a basic pose
    sample_keypoints[0, 0, 5, :] = [0.2, 0.4, 0.9]  # Left shoulder
    sample_keypoints[0, 0, 6, :] = [0.2, 0.6, 0.9]  # Right shoulder
    sample_keypoints[0, 0, 11, :] = [0.5, 0.4, 0.9]  # Left hip
    sample_keypoints[0, 0, 12, :] = [0.5, 0.6, 0.9]  # Right hip
    
    # Test comprehensive feedback
    result = feedback.get_comprehensive_feedback(sample_keypoints, 480, 640)
    
    print(f"Distance status: {result['distance_status']}")
    print(f"Person percentage: {result['person_percentage']:.1f}%")
    print(f"Visibility percentage: {result['visibility_percentage']:.1f}%")
    print(f"Message: {result['message']}")
    
    if result.get('form_analysis'):
        print(f"Form analysis available: {result['form_analysis']['phase']}")
    
    print("✓ Feedback system test completed successfully!")
    return True

def test_visualization():
    """Test the visualization system."""
    print("\nTesting Visualization System...")
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # Dark gray background
    
    # Create sample keypoints
    sample_keypoints = np.zeros((1, 1, 17, 3))
    
    # Set up a basic pose
    sample_keypoints[0, 0, 5, :] = [0.2, 0.4, 0.9]  # Left shoulder
    sample_keypoints[0, 0, 6, :] = [0.2, 0.6, 0.9]  # Right shoulder
    sample_keypoints[0, 0, 11, :] = [0.5, 0.4, 0.9]  # Left hip
    sample_keypoints[0, 0, 12, :] = [0.5, 0.6, 0.9]  # Right hip
    sample_keypoints[0, 0, 13, :] = [0.7, 0.4, 0.9]  # Left knee
    sample_keypoints[0, 0, 14, :] = [0.7, 0.6, 0.9]  # Right knee
    
    # Test enhanced visualization
    from helpers.visualization_utils import draw_prediction_on_image_enhanced
    
    try:
        result_image = draw_prediction_on_image_enhanced(test_image.copy(), sample_keypoints)
        print("✓ Enhanced visualization test completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MoveNet Lightning - Squat Analysis System Test")
    print("=" * 50)
    
    tests = [
        test_squat_analyzer,
        test_feedback_system,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The squat analysis system is ready to use.")
        print("\nTo run the system:")
        print("python main.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main() 