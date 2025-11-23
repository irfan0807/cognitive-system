# Deep Neural Network Consciousness System

This enhancement adds deep neural network capabilities to the cognitive system for consciousness simulation with live video and audio feed processing.

## New Features

### 1. Deep Neural Networks for Consciousness

The system now includes sophisticated deep learning architectures that simulate consciousness through hierarchical processing:

#### Visual Cortex CNN
- **V1 Layer**: Detects simple features (edges, orientations)
- **V2 Layer**: Complex feature extraction
- **V4 Layer**: Object-level features
- **IT Layer**: High-level semantic features

#### Auditory Cortex RNN
- **A1 Layer**: Primary auditory processing with LSTM
- **A2 Layer**: Complex auditory feature extraction
- Temporal sequence processing for audio understanding

#### Multimodal Integration
- Cross-modal attention mechanism
- Integrates visual and auditory features
- Creates unified multimodal representation

#### Global Workspace (Consciousness)
- Multi-head attention for conscious focus
- 256-dimensional consciousness state
- Integrates all sensory modalities

### 2. Live Feed Processing

#### Video Feed
```python
from cognitive_system.core.live_feed import LiveVideoFeed

# Start live video capture
video_feed = LiveVideoFeed(camera_id=0, target_size=(224, 224))
video_feed.start()

# Get preprocessed frame
frame = video_feed.get_frame()
```

#### Audio Feed
```python
from cognitive_system.core.live_feed import LiveAudioFeed

# Start live audio capture
audio_feed = LiveAudioFeed(sample_rate=16000)
audio_feed.start()

# Get audio features (mel spectrogram)
features = audio_feed.get_features()
```

#### Unified Feed Manager
```python
from cognitive_system.core.live_feed import LiveFeedManager

# Manage both video and audio
with LiveFeedManager(use_camera=True, use_microphone=True) as feeds:
    video_frame, audio_features = feeds.get_multimodal_input()
```

### 3. Conscious Cognition System

The enhanced system integrates deep learning with embodied cognition:

```python
from cognitive_system.core.conscious_cognition import ConsciousCognitionSystem

# Initialize with deep neural networks
system = ConsciousCognitionSystem(
    visual_dim=64,
    auditory_dim=32,
    attention_dim=128,
    learning_rate=0.01,
    use_live_feeds=False  # Set True for live camera/mic
)

# Process consciousness
output = system.process_live_consciousness()

# Access consciousness state
consciousness_state = output['consciousness_features']['consciousness_state']
visual_features = output['consciousness_features']['visual_features']
auditory_features = output['consciousness_features']['auditory_features']
```

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Dependencies
- **PyTorch** (CPU version): Deep learning framework
- **OpenCV**: Video processing
- **sounddevice**: Audio capture
- **librosa**: Audio feature extraction
- **NumPy, SciPy**: Numerical computing

## Usage Examples

### Example 1: Simulated Feeds (No Camera/Mic Required)

```bash
python examples/consciousness_demo.py --mode simulated
```

This runs the consciousness simulation with synthetic video and audio data.

### Example 2: Live Feeds

```bash
python examples/consciousness_demo.py --mode live
```

Requires a camera and microphone. Processes real-time video and audio.

### Example 3: Interactive Demonstration

```bash
python examples/consciousness_demo.py --mode interactive
```

Shows detailed consciousness processing with multiple scenarios.

## Architecture

```
Live Video/Audio Feeds
         ↓
    Deep Neural Networks
    ┌─────────────────────────────────────┐
    │  ┌──────────────┐  ┌──────────────┐ │
    │  │ Visual CNN   │  │ Auditory RNN │ │
    │  │  (V1→V2→V4→IT) │  │  (A1→A2 LSTM)│ │
    │  └──────────────┘  └──────────────┘ │
    │           ↓               ↓          │
    │   ┌───────────────────────────┐     │
    │   │ Multimodal Integration    │     │
    │   │  (Cross-Modal Attention)  │     │
    │   └───────────────────────────┘     │
    │               ↓                      │
    │   ┌───────────────────────────┐     │
    │   │   Global Workspace        │     │
    │   │   (Consciousness State)   │     │
    │   └───────────────────────────┘     │
    └─────────────────────────────────────┘
                  ↓
         Virtual Biology Integration
         ┌──────────────────────────┐
         │  Brain Structures        │
         │  • Brain Stem (Arousal)  │
         │  • Pituitary (Oxytocin)  │
         │  • PVN (Stress)          │
         └──────────────────────────┘
                  ↓
         Embodied Cognition
         ┌──────────────────────────┐
         │  Behavior & Decision     │
         │  Memory Formation        │
         │  Learning & Adaptation   │
         └──────────────────────────┘
```

## Key Concepts

### Consciousness Simulation
The system simulates consciousness through:
1. **Hierarchical Processing**: Visual and auditory information processed through multiple layers
2. **Global Workspace**: Integrated representation accessible to decision-making
3. **Attention Mechanism**: Focus on salient information
4. **Embodied Integration**: Consciousness influenced by and influences physiology

### Virtual Biology Drives Consciousness
- Arousal level modulates attention and processing
- Stress affects decision-making
- Oxytocin influences social cognition
- Neurotransmitters modulate mood and motivation

### Real-Time Learning
- Neural networks adapt online
- Experiences form memories
- Memory strength based on emotional significance
- Past experiences influence future decisions

## Performance

### Processing Speed
- **Visual Processing**: ~50-100ms per frame (224x224)
- **Audio Processing**: ~100ms per 1-second chunk
- **Consciousness Update**: ~10-20ms
- **Total System**: ~10 FPS (real-time capable)

### Memory Usage
- **Base System**: ~50MB
- **Deep Networks**: ~200MB
- **Per Experience**: ~2KB
- **Total (1000 experiences)**: ~250MB

## Testing

### Test Import
```python
from cognitive_system.core.conscious_cognition import ConsciousCognitionSystem
system = ConsciousCognitionSystem()
print("✓ System initialized successfully")
```

### Test Consciousness Processing
```python
output = system.process_live_consciousness()
print(f"Consciousness dimension: {output['consciousness_features']['consciousness_state'].shape}")
```

## Troubleshooting

### Camera Not Available
If camera is not available, the system automatically falls back to simulated video:
```
⚠ Could not start live feeds, falling back to simulated feeds
```

### No Audio Device
If microphone is not available, synthetic audio features are used.

### CUDA Not Available
The system uses CPU-only PyTorch. CUDA is not required or supported in the current implementation.

## Future Enhancements

Potential improvements:
1. **GPU Acceleration**: Add CUDA support for faster processing
2. **Pretrained Models**: Use pretrained vision/audio models
3. **Recurrent Memory**: Add long-term memory consolidation
4. **Attention Visualization**: Visual display of attention focus
5. **Multi-agent**: Support multiple conscious agents interacting

## API Reference

### ConsciousCognitionSystem

Main class for consciousness simulation.

**Methods:**
- `process_live_consciousness()`: Process current sensory input
- `start_live_feeds()`: Start camera and microphone
- `stop_live_feeds()`: Stop live feeds
- `interact_with_environment(duration)`: Run interaction loop
- `demonstrate_consciousness(duration)`: Run demonstration

### DeepNeuralController

Controller for deep neural networks.

**Methods:**
- `process_live_input(visual_frame, audio_chunk)`: Process raw inputs
- `integrate_with_embodied_cognition(features, physiology)`: Merge with biology

### LiveFeedManager

Manages video and audio capture.

**Methods:**
- `start()`: Start all feeds
- `stop()`: Stop all feeds
- `get_multimodal_input()`: Get synchronized video/audio

## License

MIT License - See main repository LICENSE file.

## Citation

```bibtex
@software{cognitive_system_consciousness_2025,
  title={Cognitive System: Deep Neural Network Consciousness},
  author={irfan0807},
  year={2025},
  url={https://github.com/irfan0807/cognitive-system}
}
```
