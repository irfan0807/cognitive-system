# Cognitive System - Animation with Live Video Feed

## Overview

This is your complete cognitive system application with:
- **LEFT**: Animated character responding to neural network state
- **RIGHT TOP**: Live video feed from your camera
- **RIGHT BOTTOM**: Real-time physiological metrics

## Quick Start

### Run with Camera (Recommended)
```bash
python3 run_app.py
```
or
```bash
./run.sh
```

### Run without Camera (Test Mode)
```bash
python3 run_app.py --test
```
or
```bash
./run.sh --test
```

## What You'll See

### Main Window (16:9 layout)
```
┌──────────────────────────────────────────────────────────────┐
│              COGNITIVE SYSTEM - ANIMATION WITH LIVE VIDEO    │
├────────────────────────┬────────────────────────────────────┤
│                        │                                    │
│  ANIMATED CHARACTER    │      LIVE VIDEO FEED              │
│  (responds to AI)      │      (640x480 @ 15 FPS)           │
│                        │                                    │
│  Large animation       ├────────────────────────────────────┤
│  driven by neural      │                                    │
│  network state         │  REAL-TIME METRICS                │
│                        │  • Heart Rate: 60.0 bpm           │
│                        │  • Stress Level: 0.05             │
│                        │  • Arousal: 0.50                  │
│                        │  • Mood: 0.55                     │
│                        │                                    │
└────────────────────────┴────────────────────────────────────┘
```

## Features

### Animated Character
- **Expressive**: Eyes, mouth, and body posture reflect AI state
- **Dynamic**: Limbs move based on arousal and engagement
- **Responsive**: Color and appearance change with stress levels
- **Status Display**: Shows emotional/psychological state

### Live Video Feed
- **Real-time**: 15 FPS video capture from your webcam
- **Background Processing**: Doesn't block animation
- **Thread-safe**: Uses separate thread for capture
- **Error Handling**: Gracefully falls back if camera unavailable

### Neural Network Processing
- Processes video input as sensory data
- Generates physiological responses
- Updates character animation in real-time
- Tracks emotional and cognitive states

### Physiological Metrics
- **Heart Rate**: 60-120 bpm (simulated based on AI state)
- **Stress Level**: 0-100% (affected by sensory input)
- **Arousal**: 0-100% (engagement and excitement)
- **Mood**: -1 to +1 (positive/negative emotional valence)

## Controls

| Action | Result |
|--------|--------|
| Close Window | Stop the application |
| Move in front of camera | Character responds to motion |
| - | Animation updates every frame (15 FPS) |

## System Requirements

- Python 3.7+
- macOS, Linux, or Windows
- Webcam (optional - works without it)
- 4GB RAM minimum

### Required Python Packages
```bash
pip install numpy matplotlib opencv-python torch
```

All other dependencies are already in the project.

## Architecture

```
Camera Input (15 FPS)
    ↓
┌──────────────────────────────┐
│ MediaStreamManager           │
│ (background thread)          │
└──────────┬───────────────────┘
           ↓ (frame queue)
┌──────────────────────────────┐
│ Video Display                │
│ (matplotlib imshow)          │
└──────────┬───────────────────┘
           ↓ (feature extraction)
┌──────────────────────────────┐
│ EmbodiedCognitionSystem      │
│ • Neural Network             │
│ • Virtual Nervous System     │
│ • Memory & Behavior          │
└──────────┬───────────────────┘
           ↓ (state)
┌──────────────────────────────┐
│ AnimatedPerson               │
│ • Character animation        │
│ • Metrics display            │
│ • Visual feedback            │
└──────────────────────────────┘
           ↓
        Display
```

## Troubleshooting

### No Camera Feed
**Issue**: Video feed window is empty
**Solution**: 
1. Check camera is connected
2. Try test mode: `python3 run_app.py --test`
3. Check camera permissions in system settings

### Slow Animation
**Issue**: Frame rate is choppy
**Solution**:
1. Close other applications
2. Check CPU usage: `top` or Activity Monitor
3. Try test mode (no video processing)

### Module Import Errors
**Issue**: "No module named 'cv2'" or similar
**Solution**:
```bash
pip install opencv-python
```

### Camera Permission Denied
**Issue**: Camera not opening even though connected
**Solution** (macOS):
1. System Preferences → Security & Privacy → Camera
2. Make sure Terminal/VS Code is allowed

## Performance

| Metric | Value |
|--------|-------|
| Video Capture | 15 FPS, 640x480 |
| Frame Processing | ~10ms |
| Animation FPS | 15 FPS (67ms intervals) |
| Neural Update | ~20ms |
| **Total Latency** | ~100ms camera→display |

## Files

| File | Purpose |
|------|---------|
| `run_app.py` | Main application (RECOMMENDED) |
| `run.sh` | Shell script wrapper |
| `interactive_app.py` | Alternative full-featured version |
| `test_animation.py` | Simple test without video |

## Advanced Usage

### Customize Animation Speed
Edit `run_app.py`, line ~350:
```python
FuncAnimation(..., interval=67, ...)  # milliseconds between frames
```

### Adjust Video Resolution
Edit `run_app.py`, line ~191:
```python
media_manager.add_camera_stream(name="camera", camera_id=0, fps=15)
# Change fps: 15 → 30 for higher framerate
```

### Change Neural Network Size
Edit `run_app.py`, line ~60:
```python
self.neural_network = NeuralNetworkController(
    input_size=22,      # Visual/audio features
    hidden_size=128,    # Change this: 64, 256, etc
    output_size=32      # Output neurons
)
```

## Development

### Enable Debug Logging
```bash
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from run_app import main
main()
"
```

### Test Components Separately
```python
# Test media capture
from cognitive_system.utils.media_capture import MediaStreamManager
mgr = MediaStreamManager()
mgr.add_camera_stream()
mgr.start_all()

# Test animation
from run_app import AppWithVideoFeed
app = AppWithVideoFeed(test_mode=True)
```

## Next Steps

1. **Run the app**: `python3 run_app.py`
2. **Observe**: Watch how animation responds to video
3. **Experiment**: Move, gesture, change lighting
4. **Customize**: Modify colors, metrics, or behavior
5. **Extend**: Add facial recognition, gesture detection, etc.

## Support

For issues or questions:
1. Check logs (console output)
2. Try test mode first
3. Verify camera works with: `python3 -c "import cv2; cv2.VideoCapture(0)"`
4. Review error messages for details

## License

This is part of the Cognitive System project.

---

**Ready to experience your cognitive system in action!** 🚀✨
