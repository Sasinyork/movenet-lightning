# MoveNet Lightning - Real-time Pose Detection

A powerful real-time pose detection application using Google's MoveNet Lightning and Thunder models. This project provides both webcam and video file processing capabilities with advanced features like smart cropping, temporal smoothing, and multiple model options.

## 🚀 Features

- **Real-time Webcam Processing**: Live pose detection from your camera
- **Video File Processing**: Analyze pre-recorded videos with pose detection
- **Multiple Model Options**: Choose between Lightning (fast) and Thunder (accurate)
- **Smart Cropping**: Automatic focus on detected person for better accuracy
- **Temporal Smoothing**: Reduces jitter and improves tracking stability
- **High-Quality Visualization**: Color-coded keypoints and skeleton connections
- **Progress Tracking**: Real-time progress updates for video processing
- **Output Saving**: Save processed videos with pose detection overlay

## 📋 Requirements

- Python 3.8+
- macOS (tested on macOS 24.5.0)
- Webcam access (for real-time mode)
- Sufficient storage for video processing

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd movenet-lightning
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Grant camera permissions** (macOS):
   - Go to **System Settings** > **Privacy & Security** > **Camera**
   - Enable camera access for Terminal (or your terminal app)
   - Restart Terminal after granting permissions

## 🎯 Usage

### Quick Start

Run the application:

```bash
python3 main.py
```

### Step-by-Step Guide

1. **Select Model**:

   ```
   Available Models:
   1. Lightning (faster, 192x192 input)
   2. Thunder (more accurate, 256x256 input)
   3. Lightning TFLite (optimized)
   4. Thunder TFLite (optimized)
   ```

2. **Choose Input Source**:

   ```
   1. Use Webcam (real-time)
   2. Process Video File
   ```

3. **For Webcam Mode**:

   - Position yourself in front of the camera
   - Ensure good lighting
   - Press 'q' to quit
   - Press 'r' to reset crop region

4. **For Video Mode**:
   - Provide the full path to your video file
   - Choose whether to save the processed output
   - Monitor progress during processing

## 🎮 Controls

### Webcam Mode

- **'q'**: Quit the application
- **'r'**: Reset crop region (useful if tracking gets stuck)

### Video Processing Mode

- **'q'**: Stop processing early
- **Progress updates**: Displayed every 30 frames

## 📊 Model Comparison

| Model            | Input Size | Speed    | Accuracy   | Best For         |
| ---------------- | ---------- | -------- | ---------- | ---------------- |
| Lightning        | 192×192    | ⚡⚡⚡   | ⭐⭐⭐     | Real-time webcam |
| Thunder          | 256×256    | ⚡⚡     | ⭐⭐⭐⭐⭐ | Video analysis   |
| Lightning TFLite | 192×192    | ⚡⚡⚡⚡ | ⭐⭐⭐     | Mobile/edge      |
| Thunder TFLite   | 256×256    | ⚡⚡⚡   | ⭐⭐⭐⭐⭐ | Best overall     |

## 🎨 Visualization Features

### Keypoint Colors

- **🟡 Yellow**: Head keypoints (nose, eyes, ears)
- **🟣 Magenta**: Upper body (shoulders, arms, hands)
- **🟢 Green**: Lower body (hips, knees, ankles)

### Visual Elements

- **Green Rectangle**: Crop region being analyzed
- **Colored Circles**: Keypoints with confidence-based sizing
- **Colored Lines**: Skeleton connections with confidence-based thickness

## 📁 Project Structure

```
movenet-lightning/
├── main.py                 # Main application script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── helpers/
│   ├── __init__.py
│   ├── model_utils.py     # Model loading utilities
│   ├── movenet_utils.py   # MoveNet-specific utilities
│   └── visualization_utils.py  # Visualization functions
├── data/                  # Sample data directory
└── output/               # Output directory for processed videos
```

## 🔧 Configuration

### Adjustable Parameters

In `main.py`, you can modify:

- **Smoothing Factor**: `smoothing_factor = 0.7` (0-1, higher = more smoothing)
- **Keypoint Threshold**: `keypoint_threshold=0.2` (0-1, higher = more selective)
- **Camera Resolution**: Modify `cap.set()` calls for different resolutions

### Performance Tips

1. **For Real-time Performance**:

   - Use Lightning models (options 1 or 3)
   - Ensure good lighting
   - Keep background simple

2. **For Best Accuracy**:

   - Use Thunder models (options 2 or 4)
   - Process videos instead of real-time
   - Use high-quality video input

3. **For Mobile/Edge Devices**:
   - Use TFLite models (options 3 or 4)
   - Reduce input resolution
   - Lower smoothing factor

## 🐛 Troubleshooting

### Common Issues

1. **"Cannot open webcam"**:

   - Check camera permissions in System Settings
   - Restart Terminal after granting permissions
   - Ensure no other app is using the camera

2. **Poor keypoint accuracy**:

   - Switch to Thunder model
   - Improve lighting conditions
   - Ensure full body is visible
   - Reset crop region with 'r' key

3. **Slow performance**:

   - Switch to Lightning model
   - Reduce camera resolution
   - Close other applications

4. **SSL Warnings**:
   - These are harmless warnings on macOS
   - Can be ignored or fixed by installing `urllib3==1.26.18`

### Video Format Support

Supported input formats:

- MP4, AVI, MOV, MKV, WMV, FLV
- Any resolution and frame rate
- Output maintains original properties

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google's MoveNet team for the excellent pose detection models
- TensorFlow and TensorFlow Hub for model hosting
- OpenCV community for computer vision tools

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the project structure and configuration options
3. Open an issue on the repository with detailed information

---

**Happy pose detecting! 🎯**
