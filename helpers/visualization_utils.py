# Utility functions for MoveNet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import cv2
import imageio
from IPython.display import HTML
from base64 import b64encode
# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# OpenCV colors for keypoints and edges - Enhanced for better visibility
KEYPOINT_COLOR = (255, 20, 147)  # Deep pink
EDGE_COLORS = {
    (0, 1): (255, 0, 255),   # Magenta
    (0, 2): (255, 255, 0),   # Cyan
    (1, 3): (255, 0, 255),   # Magenta
    (2, 4): (255, 255, 0),   # Cyan
    (0, 5): (255, 0, 255),   # Magenta
    (0, 6): (255, 255, 0),   # Cyan
    (5, 7): (255, 0, 255),   # Magenta
    (7, 9): (255, 0, 255),   # Magenta
    (6, 8): (255, 255, 0),   # Cyan
    (8, 10): (255, 255, 0),  # Cyan
    (5, 6): (0, 255, 255),   # Yellow
    (5, 11): (255, 0, 255),  # Magenta
    (6, 12): (255, 255, 0),  # Cyan
    (11, 12): (0, 255, 255), # Yellow
    (11, 13): (255, 0, 255), # Magenta
    (13, 15): (255, 0, 255), # Magenta
    (12, 14): (255, 255, 0), # Cyan
    (14, 16): (255, 255, 0)  # Cyan
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12), dpi=100)
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def to_gif(images, duration):
  """Converts image sequence (4D numpy array) to gif."""
  imageio.mimsave('./animation.gif', images, duration=duration)
  return embed.embed_file('./animation.gif')

def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))

def draw_prediction_on_image_enhanced(image, keypoints_with_scores, keypoint_threshold=0.15):
    """Enhanced version with improved keypoint detection and visibility for squat analysis."""
    height, width, _ = image.shape
    
    # Extract keypoints
    keypoints = keypoints_with_scores[0, 0, :, :]  # Shape: (17, 3)
    
    # Enhanced threshold system for different body parts
    # Core body parts (more important for squat analysis)
    core_keypoints = [5, 6, 11, 12, 13, 14]  # Shoulders, hips, knees
    # Upper body parts
    upper_body_keypoints = [0, 1, 2, 3, 4, 7, 8, 9, 10]  # Head, arms
    # Lower body parts
    lower_body_keypoints = [15, 16]  # Ankles
    
    # Draw keypoints with enhanced visibility
    for i in range(17):
        confidence = keypoints[i, 2]
        
        # Use different thresholds for different body parts
        if i in core_keypoints:
            # Lower threshold for core parts (more important for squat analysis)
            threshold = keypoint_threshold * 0.8
        elif i in upper_body_keypoints:
            # Medium threshold for upper body
            threshold = keypoint_threshold * 0.9
        else:
            # Normal threshold for extremities
            threshold = keypoint_threshold
        
        if confidence > threshold:
            x_norm = keypoints[i, 1]
            y_norm = keypoints[i, 0]
            
            # Convert to pixel coordinates
            x = int(x_norm * width)
            y = int(y_norm * height)
            
            # Allow keypoints slightly outside bounds (for raised arms, etc.)
            x = max(-15, min(width + 15, x))
            y = max(-15, min(height + 15, y))
            
            # Enhanced circle size based on confidence and body part importance
            if i in core_keypoints:
                radius = int(4 + confidence * 6)  # Larger for core parts
                thickness = 3
            elif i in upper_body_keypoints:
                radius = int(3 + confidence * 4)  # Medium for upper body
                thickness = 2
            else:
                radius = int(2 + confidence * 3)  # Smaller for extremities
                thickness = 1
            
            # Enhanced color coding based on body part importance for squat analysis
            if i in [0, 1, 2, 3, 4]:  # Head
                color = (0, 255, 255)  # Yellow
            elif i in [5, 6]:  # Shoulders (critical for back analysis)
                color = (255, 0, 0)  # Red
            elif i in [7, 8, 9, 10]:  # Arms
                color = (255, 0, 255)  # Magenta
            elif i in [11, 12]:  # Hips (critical for squat analysis)
                color = (0, 255, 0)  # Green
            elif i in [13, 14]:  # Knees (critical for squat analysis)
                color = (255, 165, 0)  # Orange
            elif i in [15, 16]:  # Ankles
                color = (128, 0, 128)  # Purple
            else:
                color = (255, 20, 147)  # Deep pink
            
            # Only draw if keypoint is reasonably within bounds
            if 0 <= x < width and 0 <= y < height:
                # Draw filled circle with border
                cv2.circle(image, (x, y), radius, color, -1)
                cv2.circle(image, (x, y), radius, (255, 255, 255), thickness)
                
                # Add confidence indicator for core parts
                if i in core_keypoints and confidence > 0.7:
                    cv2.circle(image, (x, y), radius + 2, (0, 255, 0), 1)
    
    # Draw edges with enhanced visibility
    for edge_pair, color in EDGE_COLORS.items():
        confidence1 = keypoints[edge_pair[0], 2]
        confidence2 = keypoints[edge_pair[1], 2]
        
        # Use adaptive thresholds for edges
        if edge_pair[0] in core_keypoints or edge_pair[1] in core_keypoints:
            threshold = keypoint_threshold * 0.8
        elif edge_pair[0] in upper_body_keypoints or edge_pair[1] in upper_body_keypoints:
            threshold = keypoint_threshold * 0.9
        else:
            threshold = keypoint_threshold
        
        if confidence1 > threshold and confidence2 > threshold:
            x1_norm = keypoints[edge_pair[0], 1]
            y1_norm = keypoints[edge_pair[0], 0]
            x2_norm = keypoints[edge_pair[1], 1]
            y2_norm = keypoints[edge_pair[1], 0]
            
            x1 = int(x1_norm * width)
            y1 = int(y1_norm * height)
            x2 = int(x2_norm * width)
            y2 = int(y2_norm * height)
            
            # Allow edges to extend slightly outside bounds
            x1 = max(-15, min(width + 15, x1))
            y1 = max(-15, min(height + 15, y1))
            x2 = max(-15, min(width + 15, x2))
            y2 = max(-15, min(height + 15, y2))
            
            # Only draw edge if at least one endpoint is within bounds
            if (0 <= x1 < width and 0 <= y1 < height) or (0 <= x2 < width and 0 <= y2 < height):
                # Enhanced thickness based on confidence and importance
                avg_confidence = (confidence1 + confidence2) / 2
                if edge_pair[0] in core_keypoints or edge_pair[1] in core_keypoints:
                    thickness = max(3, int(avg_confidence * 5))
                else:
                    thickness = max(2, int(avg_confidence * 3))
                
                # Draw the edge
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                
                # Add subtle glow effect for core connections
                if edge_pair[0] in core_keypoints or edge_pair[1] in core_keypoints:
                    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    return image

def draw_prediction_on_image_adaptive(image, keypoints_with_scores, keypoint_threshold=0.15):
    """Enhanced version that handles close range and raised arms better."""
    height, width, _ = image.shape
    
    # Extract keypoints
    keypoints = keypoints_with_scores[0, 0, :, :]  # Shape: (17, 3)
    
    # Adaptive threshold based on keypoint visibility
    # Lower threshold for extremities (hands, feet) since they're more likely to be cut off
    extremity_keypoints = [9, 10, 15, 16]  # Left/right wrists and ankles
    upper_body_keypoints = [5, 6, 7, 8]    # Shoulders and elbows
    
    # Draw keypoints with adaptive thresholds
    for i in range(17):
        confidence = keypoints[i, 2]
        
        # Use different thresholds for different body parts
        if i in extremity_keypoints:
            # Lower threshold for hands/feet (they might be cut off)
            threshold = keypoint_threshold * 0.7
        elif i in upper_body_keypoints:
            # Medium threshold for upper body
            threshold = keypoint_threshold * 0.9
        else:
            # Normal threshold for core body parts
            threshold = keypoint_threshold
        
        if confidence > threshold:
            x_norm = keypoints[i, 1]
            y_norm = keypoints[i, 0]
            
            # Convert to pixel coordinates
            x = int(x_norm * width)
            y = int(y_norm * height)
            
            # Allow keypoints slightly outside bounds (for raised arms, etc.)
            # But clamp them to reasonable limits
            x = max(-10, min(width + 10, x))
            y = max(-10, min(height + 10, y))
            
            # Adjust circle size based on confidence and body part
            if i in extremity_keypoints:
                radius = int(2 + confidence * 3)  # Smaller for extremities
            else:
                radius = int(3 + confidence * 4)  # Normal size for core parts
            
            # Color coding based on body part
            if i in [0, 1, 2, 3, 4]:  # Head
                color = (0, 255, 255)  # Yellow
            elif i in [5, 6, 7, 8, 9, 10]:  # Upper body
                color = (255, 0, 255)  # Magenta
            elif i in [11, 12, 13, 14, 15, 16]:  # Lower body
                color = (0, 255, 0)  # Green
            else:
                color = (255, 20, 147)  # Deep pink
            
            # Only draw if keypoint is reasonably within bounds
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(image, (x, y), radius, color, -1)
                cv2.circle(image, (x, y), radius, (255, 255, 255), 1)
    
    # Draw edges with adaptive logic
    for edge_pair, color in EDGE_COLORS.items():
        confidence1 = keypoints[edge_pair[0], 2]
        confidence2 = keypoints[edge_pair[1], 2]
        
        # Use adaptive thresholds for edges too
        if edge_pair[0] in extremity_keypoints or edge_pair[1] in extremity_keypoints:
            threshold = keypoint_threshold * 0.7
        elif edge_pair[0] in upper_body_keypoints or edge_pair[1] in upper_body_keypoints:
            threshold = keypoint_threshold * 0.9
        else:
            threshold = keypoint_threshold
        
        if confidence1 > threshold and confidence2 > threshold:
            x1_norm = keypoints[edge_pair[0], 1]
            y1_norm = keypoints[edge_pair[0], 0]
            x2_norm = keypoints[edge_pair[1], 1]
            y2_norm = keypoints[edge_pair[1], 0]
            
            x1 = int(x1_norm * width)
            y1 = int(y1_norm * height)
            x2 = int(x2_norm * width)
            y2 = int(y2_norm * height)
            
            # Allow edges to extend slightly outside bounds
            x1 = max(-10, min(width + 10, x1))
            y1 = max(-10, min(height + 10, y1))
            x2 = max(-10, min(width + 10, x2))
            y2 = max(-10, min(height + 10, y2))
            
            # Only draw edge if at least one endpoint is within bounds
            if (0 <= x1 < width and 0 <= y1 < height) or (0 <= x2 < width and 0 <= y2 < height):
                thickness = max(2, int((confidence1 + confidence2) / 2 * 3))
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    
    return image

def draw_prediction_on_image_simple(image, keypoints_with_scores, keypoint_threshold=0.15):
    """Simple version optimized for close-medium range."""
    height, width, _ = image.shape
    
    # Extract keypoints
    keypoints = keypoints_with_scores[0, 0, :, :]  # Shape: (17, 3)
    
    # Draw keypoints
    for i in range(17):
        confidence = keypoints[i, 2]
        if confidence > keypoint_threshold:
            x_norm = keypoints[i, 1]
            y_norm = keypoints[i, 0]
            
            # Convert to pixel coordinates
            x = int(x_norm * width)
            y = int(y_norm * height)
            
            # Allow slight overflow for raised arms
            x = max(-5, min(width + 5, x))
            y = max(-5, min(height + 5, y))
            
            # Adjust circle size based on confidence
            radius = int(3 + confidence * 4)
            
            # Use different colors for different body parts
            if i in [0, 1, 2, 3, 4]:  # Head keypoints
                color = (0, 255, 255)  # Yellow
            elif i in [5, 6, 7, 8, 9, 10]:  # Upper body
                color = (255, 0, 255)  # Magenta
            elif i in [11, 12, 13, 14, 15, 16]:  # Lower body
                color = (0, 255, 0)  # Green
            else:
                color = (255, 20, 147)  # Deep pink
            
            # Only draw if keypoint is within reasonable bounds
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(image, (x, y), radius, color, -1)
                cv2.circle(image, (x, y), radius, (255, 255, 255), 1)
    
    # Draw edges
    for edge_pair, color in EDGE_COLORS.items():
        confidence1 = keypoints[edge_pair[0], 2]
        confidence2 = keypoints[edge_pair[1], 2]
        
        if confidence1 > keypoint_threshold and confidence2 > keypoint_threshold:
            x1_norm = keypoints[edge_pair[0], 1]
            y1_norm = keypoints[edge_pair[0], 0]
            x2_norm = keypoints[edge_pair[1], 1]
            y2_norm = keypoints[edge_pair[1], 0]
            
            x1 = int(x1_norm * width)
            y1 = int(y1_norm * height)
            x2 = int(x2_norm * width)
            y2 = int(y2_norm * height)
            
            # Allow slight overflow
            x1 = max(-5, min(width + 5, x1))
            y1 = max(-5, min(height + 5, y1))
            x2 = max(-5, min(width + 5, x2))
            y2 = max(-5, min(height + 5, y2))
            
            # Only draw if at least one endpoint is visible
            if (0 <= x1 < width and 0 <= y1 < height) or (0 <= x2 < width and 0 <= y2 < height):
                thickness = max(2, int((confidence1 + confidence2) / 2 * 3))
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    
    return image