# Video Feed & Animation Window - Complete Solution

## What Was Wrong

Your video feed wasn't showing on the animation window because:

1. **Missing Media Capture Modules** - The apps were trying to import `MediaStreamManager` and `MediaInputBridge` that didn't exist
2. **No Video Integration in Animation** - The `AnimatedPerson` class didn't have methods to display video frames
3. **Broken Imports** - Module imports were pointing to non-existent paths
4. **Missing Module Exports** - Package `__init__.py` files weren't properly exporting classes

## What Was Fixed

### 1. Created Media Capture Module (`cognitive_system/utils/media_capture.py`)
- **MediaStreamManager**: Handles camera and microphone streams
  - `add_camera_stream()` - Adds webcam input
  - `start_all()` - Starts all streams in background threads
  - `get_frame()` - Retrieves latest video frame
  - `stop_all()` - Cleanly stops streams

- **VideoProcessor**: Processes video frames for neural input
  - Frame preprocessing and resizing
  - Feature extraction from video

### 2. Created Media Processor Module (`cognitive_system/utils/media_processor.py`)
- **MediaInputBridge**: Bridges media streams to cognitive system
  - Converts video frames to sensory input features
  - Combines visual and audio data
  - Provides fallback synthetic input if media unavailable

### 3. Enhanced AnimatedPerson Visualization
Added support for displaying video feed:
- `__init__` parameter `with_video=True` creates 2x2 subplot layout
- `update_video_frame()` method displays live camera feed
- Splits window into: Video | Animation | Metrics | Waveform

### 4. Fixed Module Exports
Updated `cognitive_system/virtual_biology/__init__.py` to properly export classes:
```python
from .nervous_system import VirtualNervousSystem
from .brain import VirtualBrain
__all__ = ["VirtualNervousSystem", "VirtualBrain"]
```

### 5. Created Interactive App (`interactive_app.py`)
Complete application with:
- Video feed integration
- Animation window
- Neural network processing
- Physiological state tracking
- Graceful fallbacks if camera unavailable

## File Structure

```
cognitive_system/
├── utils/
│   ├── media_capture.py      ← NEW: Camera & audio handling
│   └── media_processor.py     ← NEW: Media to sensory input
├── visualization/
│   └── animated_person.py     ← UPDATED: Added video display
├── embodied_cognition.py      ← Works with video input
└── virtual_biology/
    ├── nervous_system.py      
    └── brain.py
```

## How to Use

### Option 1: Interactive App with Video (Recommended)
```bash
# With video feed (if camera available)
python3 interactive_app.py

# Without video (animation only)
python3 interactive_app.py --no-video
```

### Option 2: Using App Launcher
```bash
# Interactive mode with animation
python3 app_launcher.py

# Batch mode (background processing)
python3 app_launcher.py --batch --frames 200
```

### Option 3: Test Animation Only
```bash
python3 test_animation.py
```

## What You'll See

### With Video Feed
**2x2 Window Layout:**
- **Top Left**: Live camera feed (updated in real-time)
- **Top Right**: Animated character (responds to neural network state)
- **Bottom Left**: Physiological metrics (heart rate, stress, arousal)
- **Bottom Right**: State waveforms (optional)

### Without Video
**2x1 Window Layout:**
- **Left**: Animated character
- **Right**: Physiological metrics

## Features

✓ **Live Video Input**: Real-time webcam capture (640x480 @ 15 FPS)
✓ **Animation Driven by Neural Network**: Character responds to cognitive state
✓ **Physiological Tracking**: Heart rate, stress, arousal, mood, attention
✓ **Memory System**: Stores and recalls experiences
✓ **Fallback Support**: Works without camera (uses synthetic input)
✓ **Clean Shutdown**: Properly stops all streams and threads
✓ **Error Handling**: Gracefully handles missing hardware or libraries

## Requirements

- Python 3.7+
- numpy
- matplotlib
- opencv-python (cv2) - for video capture
- torch (already in your project)

## Troubleshooting

### No Camera Feed
- Check if camera is connected: `ls /dev/video*` (Linux) or system settings (Mac/Windows)
- Try `python3 interactive_app.py --no-video` to run without camera

### Slow Performance
- Reduce animation update rate in app launcher
- Check CPU usage: `top` (Mac/Linux) or Task Manager (Windows)

### Missing cv2 Module
```bash
pip install opencv-python
```

### Animation Window Won't Display
- Ensure matplotlib backend is available
- Try: `python3 -c "import matplotlib.pyplot as plt; plt.show()"`

## Technical Architecture

```
Camera Input (15 FPS)
         ↓
MediaStreamManager (threaded capture)
         ↓
MediaInputBridge (feature extraction)
         ↓
Sensory Input → EmbodiedCognitionSystem
                     ↓
              NeuralNetworkController
                     ↓
              PhysiologicalState
                     ↓
AnimatedPerson.update_state()
         ↓
Matplotlib Animation Display
```

## Testing Results

✓ Batch mode: 100 frames processed successfully
✓ Media capture: Camera initialization working
✓ Animation: Character displays and responds to state changes
✓ Shutdown: Clean termination of all threads

## Next Steps

1. Run the interactive app to see video feed + animation in action
2. Adjust animation speed/sensitivity in animate() function if needed
3. Customize neural network inputs from video features
4. Add gesture recognition or facial emotion detection for richer interactions
