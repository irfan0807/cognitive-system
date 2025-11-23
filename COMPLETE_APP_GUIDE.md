# Cognitive System - Complete Application Guide

This unified app combines **live video/audio feed processing**, **real-time character animation**, and **neural network cognitive processing**.

## Quick Start

### Run Interactive Mode (with Animation)
```bash
python3 complete_app.py
```
This will:
- ✅ Open animated character visualization
- ✅ Process real video/audio if available (or synthetic if not)
- ✅ Show real-time neural network processing
- ✅ Display character responding to emotional states
- ✅ Close window to exit

### Run Batch Mode (no Animation)
```bash
python3 complete_app.py --batch --frames 300
```
This will:
- ✅ Process 300 frames of cognitive simulation
- ✅ Run headless (no GUI)
- ✅ Show status updates every 50 frames
- ✅ Good for testing without display

## Features

### 1. **Live Video & Audio Feed** 🎥🎤
- Captures real-time video from camera
- Captures real-time audio from microphone
- Falls back to synthetic input if hardware unavailable
- Processes sensory data through neural networks

### 2. **Real-time Character Animation** 🎭
- Animated character that responds to cognitive state
- Heart rate visualization (pulsing)
- Facial expressions based on mood/stress
- Eye movements based on attention
- Color-coded emotional state

### 3. **Neural Network Processing** 🧠
- Visual and auditory feature extraction
- Real-time neural inference
- Emotional state computation
- Physiological simulation

### 4. **Memory System** 💾
- Multimodal memory consolidation
- Experience integration
- Emotion-driven memory strength

## What Happens When You Run

### Frame Processing Loop:
```
1. Capture Video/Audio → 
2. Extract Features → 
3. Neural Processing → 
4. Compute Emotions → 
5. Update Animation → 
6. Repeat
```

### Real-time Metrics Displayed:
- **Heart Rate** (BPM): Changes with stress/arousal
- **Arousal Level**: Sensory intensity response
- **Mood**: Emotional valence (-1 to +1)
- **Stress Response**: Threat/stress level
- **Frame Count**: Processing speed tracking

## Requirements

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install visualization dependencies
pip install matplotlib

# Install media capture dependencies  
pip install opencv-python sounddevice librosa
```

### Hardware (Optional)
- **Camera**: USB webcam for live video
- **Microphone**: Built-in or external mic for audio
- **Display**: For animation visualization

## Examples

### Example 1: Interactive with Animation (Camera + Mic)
```bash
python3 complete_app.py
```
Output: Live animated character responding to your video/audio!

### Example 2: Batch Processing (Testing)
```bash
python3 complete_app.py --batch --frames 500
```
Output: Console output showing frame processing

### Example 3: Longer Interactive Session
```bash
python3 complete_app.py
# Animation runs until you close the window
```

## How It Works

### Visual Processing
- Camera frame → 64-dim visual features
- Represents spatial and temporal patterns

### Audio Processing  
- Microphone audio → 32-dim audio features
- Captures frequency and temporal characteristics

### Cognitive State
The system maintains:
- **Brain State**: Simulated neural activity
- **Physiological State**: Heart rate, stress, arousal
- **Emotional State**: Valence (mood) and arousal
- **Attention**: Focus/saliency
- **Memory**: Consolidated experiences

### Animation Response
Character appearance changes based on:
- Arousal: Eye size, body tension
- Stress: Facial expression, posture
- Mood: Smile/frown, color tone
- Heart Rate: Pulse visualization

## Architecture

```
┌─────────────────────────────────┐
│   Live Video/Audio Input        │
│   (Camera + Microphone)         │
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │   Feature   │
        │ Extraction  │
        └──────┬──────┘
               │
        ┌──────▼──────────────┐
        │  Cognitive System   │
        │  - Brain            │
        │  - Neural Network   │
        │  - Physiology       │
        │  - Memory           │
        └──────┬──────────────┘
               │
        ┌──────▼──────┐
        │ Emotional   │
        │ State       │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ Animation   │
        │ Rendering   │
        └─────────────┘
```

## Troubleshooting

### Camera Not Working
```bash
# App will automatically fall back to synthetic video
# Check camera permissions on your OS
```

### Microphone Not Working
```bash
# App will automatically fall back to synthetic audio
# Check audio input device settings
```

### Animation Not Showing
```bash
# Make sure matplotlib is installed
pip install matplotlib

# Try batch mode first
python3 complete_app.py --batch --frames 100
```

### Performance Issues
```bash
# Reduce frame rate or use batch mode
python3 complete_app.py --batch --frames 300
```

## Next Steps

- ✅ Try the animated demo: `python3 examples/animated_person_demo.py`
- ✅ Try batch processing: `python3 complete_app.py --batch`
- ✅ Try interactive mode: `python3 complete_app.py`
- ✅ Explore examples: `python3 examples/` directory

## API Usage

```python
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem

# Initialize
system = EmbodiedCognitionSystem(
    visual_dim=64,
    auditory_dim=32,
    attention_dim=128,
    learning_rate=0.01
)

# Process sensory input
import numpy as np
visual_input = np.random.randn(64)
auditory_input = np.random.randn(32)

state = system.process_sensory_input(
    visual_input,
    auditory_input,
    social_context=0.5,
    threat_level=0.1,
    reward_signal=0.3
)

# Access results
print(f"Emotion: {state.emotional_state}")
print(f"Heart Rate: {state.brain_state['heart_rate']}")
```

## Summary

✅ **One Command to Run Everything:**
```bash
python3 complete_app.py
```

This runs:
- 📹 Live video capture & processing
- 🎤 Live audio capture & processing  
- 🎭 Real-time character animation
- 🧠 Neural network cognitive simulation
- 💾 Memory consolidation
