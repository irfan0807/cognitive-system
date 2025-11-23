# Quick Start - Video Feed Animation Window

## Your Video Feed Issue - SOLVED ✓

The video feed wasn't showing because the system was missing critical components:
- Media capture modules for camera input
- Video display integration in animation
- Proper module initialization

**Everything is now fixed!**

## Get Started in 30 Seconds

### 1. Simple Animation (No Video) - Fastest
```bash
python3 test_animation.py
```
✓ Shows animated character responding to simulated neural states

### 2. Full System - Batch Processing
```bash
python3 app_launcher.py --batch --frames 200
```
✓ Processes 200 frames with video feed capture (runs in background)

### 3. Interactive App - Video + Animation (RECOMMENDED)
```bash
python3 interactive_app.py
```
✓ Live video feed + animated character in split-screen window
✓ Shows physiological state in real-time
✓ Close window to exit

### 4. Without Camera (If No Webcam)
```bash
python3 interactive_app.py --no-video
```
✓ Animation only mode
✓ Uses simulated sensory input

## What You're Seeing

### Video Feed Window Layout
```
┌─────────────────────┬──────────────────────┐
│                     │                      │
│   Live Camera       │  Animated Person    │
│   Feed              │  (responds to AI)    │
│                     │                      │
├─────────────────────┼──────────────────────┤
│                     │                      │
│  Neural Network     │  Heart Rate,        │
│  Metrics            │  Stress, Arousal    │
│                     │                      │
└─────────────────────┴──────────────────────┘
```

## New Files Created

✓ `cognitive_system/utils/media_capture.py` - Camera handling
✓ `cognitive_system/utils/media_processor.py` - Video to neural input  
✓ `interactive_app.py` - Complete interactive application
✓ `cognitive_system/visualization/` updated - Video display support

## System Now Supports

✓ **Live Video Input** - Real-time webcam capture
✓ **Neural Processing** - Video → Features → Neural Network
✓ **Animated Responses** - Character reacts to AI state
✓ **Physiological Tracking** - Heart rate, stress, arousal
✓ **Clean Shutdown** - Safe thread termination
✓ **Graceful Fallbacks** - Works with or without camera

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'cv2'" | `pip install opencv-python` |
| Camera not opening | Check camera is connected & not in use |
| Slow animation | Close other apps, or use `--no-video` mode |
| Window won't display | Ensure matplotlib backend available |

## Test Results ✓

- ✓ Batch mode: 100 frames processed
- ✓ Animated person: Character displays and animates
- ✓ Media capture: Camera detected and capturing
- ✓ Neural network: Processing sensory input
- ✓ Memory system: Storing experiences

## Architecture

```
┌─────────────────────────────────────┐
│     LIVE VIDEO FEED (Webcam)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    MediaStreamManager (threads)     │
│    ├─ Camera capture (15 FPS)       │
│    └─ Feature extraction            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Cognitive System (Neural Network)  │
│   ├─ Visual processing              │
│   ├─ Embodied cognition             │
│   └─ Physiological state            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   AnimatedPerson (Visualization)    │
│   ├─ Video display                  │
│   ├─ Character animation            │
│   └─ Metrics display                │
└─────────────────────────────────────┘
```

## Next Steps

1. **Run interactive app**: `python3 interactive_app.py`
2. **Watch the animation** respond to video input
3. **Experiment**: Move in front of camera, observe character response
4. **Customize**: Modify animation colors/behavior in `animated_person.py`
5. **Extend**: Add gesture recognition or facial emotion detection

## Questions?

- Check logs for detailed information
- All components have logging enabled
- Errors show up in the terminal with full stack traces

---

**Your cognitive system is now fully functional with video feed support!** 🎥✨
