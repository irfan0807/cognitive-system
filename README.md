# Cognitive System - Embodied Cognition AI with Deep Neural Networks

An autonomous AI system with **embodied cognition**, driven by **deep neural networks** and a **simplified brain simulation model** for **real-time learning** and **consciousness simulation**. The system includes a **complete virtual nervous system** and **physiology** where **virtual biology drives virtual cognition**, enhanced with **live video and audio feed processing**.

## 🆕 NEW: Terminal Interface - Interactive AI Chat

**Experience the AI through a retro green terminal!** The system can now:
- 👁️ See you through your camera (displayed as ASCII art)
- 🗣️ Talk to you using text-to-speech
- 💚 Beautiful green-on-black terminal theme
- 🧠 Real-time state monitoring

**Quick Start:**
```bash
python terminal_app.py
```

See [Terminal Quick Start Guide](TERMINAL_QUICK_START.md) for details.

## 🆕 NEW: Deep Neural Network Consciousness

The system now includes advanced deep learning capabilities for consciousness simulation:

- **Deep CNN Visual Cortex**: Hierarchical processing (V1→V2→V4→IT layers)
- **LSTM Auditory Cortex**: Temporal audio sequence processing
- **Multimodal Integration**: Cross-modal attention and fusion
- **Global Workspace**: Consciousness state representation
- **Live Feed Processing**: Real-time video and audio capture
- **Embodied Integration**: Deep learning merged with virtual biology

See [Deep Neural Networks Documentation](docs/DEEP_NEURAL_NETWORKS.md) for details.

## Overview

This system implements an advanced cognitive architecture that integrates:

- **Virtual Brain Structures**
  - Brain Stem: Controls basic life functions and arousal
  - Pituitary Gland: Regulates oxytocin for social bonding
  - Paraventricular Nucleus (PVN): Manages stress response

- **Neural Networks for Real-time Learning**
  - Online learning capabilities
  - Attention mechanisms
  - Decision-making networks

- **Multimodal Memory System**
  - Integrates emotion, visual, and auditory stimuli
  - Memory consolidation based on emotional significance
  - Context-aware retrieval

- **Virtual Nervous System & Physiology**
  - Autonomic nervous system (sympathetic/parasympathetic)
  - Neurotransmitter modulation (dopamine, serotonin, norepinephrine, GABA)
  - Homeostatic regulation

- **🆕 Deep Neural Networks for Consciousness**
  - Visual cortex CNN with hierarchical layers
  - Auditory cortex RNN with temporal processing
  - Multimodal integration network
  - Global workspace consciousness mechanism
  - Live video and audio feed processing

- **RAG System for Live Video/Audio** 🆕
  - Real-time processing of video and audio streams
  - Multimodal embedding extraction and fusion
  - Vector-based memory retrieval for context-aware decisions
  - Real-time cognitive behavior visualization

## Key Features

### 1. Virtual Biology Drives Virtual Cognition

The system demonstrates how biological processes influence cognitive functions:

- **Arousal** from the brain stem modulates attention and heart rate
- **Oxytocin** from the pituitary gland promotes social bonding and buffers stress
- **Stress responses** from the PVN affect decision-making
- **Neurotransmitters** modulate mood, motivation, attention, and calmness
- **Physiological state** (energy, homeostasis) influences behavior

### 2. Multimodal Memory Integration

Memories are formed by integrating:
- Visual features
- Auditory features  
- Emotional valence and arousal
- Physiological state at the time
- Contextual information

Emotionally salient experiences are consolidated more strongly, mimicking biological memory processes.

### 3. Real-time Learning

Neural networks adapt online through:
- Continuous sensory processing
- Reward-based learning
- Experience-driven weight updates

### 4. RAG System for Live Streams 🆕

The system includes a state-of-the-art RAG (Retrieval-Augmented Generation) system:

- **Live Video/Audio Processing**: Extracts features from live streams in real-time
- **Vector-based Memory**: Stores multimodal embeddings for efficient retrieval
- **Context-Aware Decisions**: Retrieves relevant past experiences to inform current decisions
- **Cognitive Visualization**: Real-time display of attention, emotions, physiology, and decision-making
- **Multimodal Fusion**: Combines visual and audio information with attention weighting

See [RAG System Documentation](docs/RAG_SYSTEM.md) for detailed information.

## Installation

```bash
# Clone the repository
git clone https://github.com/irfan0807/cognitive-system.git
cd cognitive-system

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem
import numpy as np

# Initialize the system
system = EmbodiedCognitionSystem(
    visual_dim=64,
    auditory_dim=32,
    attention_dim=128,
    learning_rate=0.01
)

# Process sensory input
visual_input = np.random.randn(64) * 0.5
auditory_input = np.random.randn(32) * 0.5

state = system.process_sensory_input(
    visual_input=visual_input,
    auditory_input=auditory_input,
    social_context=0.8,  # Social interaction level
    threat_level=0.1,    # Threat perception
    reward_signal=0.6    # Positive reinforcement
)

# Get current state
summary = system.get_state_summary()
print(f"Arousal: {summary['arousal']:.3f}")
print(f"Oxytocin: {summary['oxytocin']:.3f}")
print(f"Stress: {summary['stress']:.3f}")
print(f"Mood: {summary['mood']:.3f}")

# Make a decision
actions = ["approach", "avoid", "explore", "rest"]
action, confidence = system.make_decision(actions)
print(f"Selected action: {action} (confidence: {confidence:.3f})")
```

### Deep Neural Network Consciousness

```python
from cognitive_system.core.conscious_cognition import ConsciousCognitionSystem

# Initialize system with deep neural networks
system = ConsciousCognitionSystem(
    visual_dim=64,
    auditory_dim=32,
    attention_dim=128,
    learning_rate=0.01,
    use_live_feeds=False  # Set True for camera/microphone
)

# Process live consciousness (uses simulated or real feeds)
output = system.process_live_consciousness()

# Access consciousness state
consciousness = output['consciousness_features']['consciousness_state']
print(f"Consciousness state dimension: {consciousness.shape}")

# See physiological integration
physiology = output['physiological_state']
print(f"Arousal: {physiology['arousal']:.3f}")
print(f"Heart rate: {physiology['heart_rate']:.1f} bpm")
print(f"Stress: {physiology['stress']:.3f}")

# Run interactive demonstration
system.demonstrate_consciousness(duration=10.0)
```

## Running the Demo

### Basic Embodied Cognition Demo
```bash
python examples/demo.py
```

### 🆕 Deep Neural Network Consciousness Demo
```bash
# Simulated feeds (no camera/mic required)
python examples/consciousness_demo.py --mode simulated

# Live feeds (requires camera/mic)
python examples/consciousness_demo.py --mode live

# Interactive detailed demonstration
python examples/consciousness_demo.py --mode interactive
```

This will run demonstrations showing:
- Calm, positive social interaction
- Stressful, threatening situation
- Social buffering (how social support reduces stress)
- Decision-making influenced by biological state
- Memory formation and retrieval
- Integration of biology and cognition

### RAG System Demo 🆕

```bash
python examples/rag_live_stream_demo.py
```

This demonstrates the RAG system with live video/audio processing:
- Processing different scene types (calm, active, complex)
- Different audio types (speech, music, noise, ambient)
- Real-time cognitive behavior visualization
- RAG-based context retrieval and decision-making
- Multimodal memory formation and retrieval

### 🆕 Terminal Interface - Interactive AI Chat

Interact with the AI through a retro green-on-black terminal interface:

```bash
# Full mode with camera and speech
python terminal_app.py

# Camera only (no speech)
python terminal_app.py --no-speech

# Text only (no camera)
python terminal_app.py --no-camera
```

The terminal interface features:
- Live camera feed displayed as ASCII art in the terminal
- AI sees you and responds contextually
- Text-to-speech for AI responses
- Real-time display of arousal, mood, stress, heart rate, and attention
- Retro green-on-black terminal theme
- Conversation history

See [Terminal Interface Documentation](TERMINAL_INTERFACE_README.md) for details.

### Animated Person Visualization

See the neural network in action with an animated character that responds to cognitive states:

```bash
# Generate static visualizations
python examples/animated_person_simple_demo.py

# Run full animation (requires display)
python examples/animated_person_demo.py
```

The animated person shows:
- Character appearance changes based on emotional state (color reflects valence)
- Body movements driven by arousal and stress levels
- Real-time heart rate pulsing visualization
- Neural network metrics displayed in real-time

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Sensory Input Layer                      │
│              (Visual, Auditory, Social, Threat)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Virtual Brain Structures                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Brain Stem  │  │  Pituitary   │  │ Paraventricular  │  │
│  │  (Arousal)  │  │  (Oxytocin)  │  │   Nucleus (PVN)  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Virtual Nervous System & Physiology            │
│  ┌────────────────────────┐  ┌──────────────────────────┐  │
│  │  Autonomic System      │  │  Neurotransmitters       │  │
│  │  (Sympathetic/Para)    │  │  (DA, 5HT, NE, GABA)     │  │
│  └────────────────────────┘  └──────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Cognitive Processing Layer                 │
│  ┌────────────────────────┐  ┌──────────────────────────┐  │
│  │  Attention Network     │  │  Decision Network        │  │
│  │  (Real-time Learning)  │  │  (Biology-driven)        │  │
│  └────────────────────────┘  └──────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Multimodal Memory System                   │
│         (Emotion + Visual + Auditory Integration)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Behavioral Output                        │
│              (Actions, Decisions, Learning)                 │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Brain Module (`cognitive_system.brain`)
- `BrainStem`: Basic life functions and arousal control
- `PituitaryGland`: Oxytocin regulation for social bonding
- `ParaventricularNucleus`: Stress response and hormonal integration
- `VirtualBrain`: Integrates all brain structures

### Neural Module (`cognitive_system.neural`)
- `NeuralLayer`: Individual neural network layer with online learning
- `ActivationFunction`: Sigmoid, tanh, ReLU activations
- `RealtimeNeuralNetwork`: Multi-layer network for real-time learning

### Memory Module (`cognitive_system.memory`)
- `MultimodalMemory`: Single memory integrating multiple modalities
- `MemoryConsolidation`: Strengthens emotionally significant memories
- `MultimodalMemorySystem`: Complete memory storage and retrieval

### Physiology Module (`cognitive_system.physiology`)
- `Neurotransmitter`: Models neurotransmitter dynamics
- `AutonomicNervousSystem`: Sympathetic/parasympathetic balance
- `VirtualNervousSystem`: Complete nervous system simulation
- `VirtualPhysiology`: Integrated physiological processes

### Visualization Module (`cognitive_system.visualization`)
- `AnimatedPerson`: Visual representation of a character connected to the neural network
  - Real-time animation driven by neural network states
  - Emotional expression through color and movement
  - Physiological indicators (heart rate, breathing)
  - Metrics dashboard showing all neural network parameters

### Embodied Cognition (`cognitive_system.embodied_cognition`)
- `EmbodiedCognitionSystem`: Main integration where biology drives cognition

### RAG Module (`cognitive_system.rag`) 🆕
- `MultimodalRAGSystem`: Complete RAG system for live video/audio
- `VectorStore`: Efficient vector database for similarity search
- `VideoStreamProcessor`: Extracts features from video frames
- `AudioStreamProcessor`: Extracts features from audio chunks
- `CognitiveVisualizer`: Real-time cognitive behavior visualization

See [RAG System Documentation](docs/RAG_SYSTEM.md) for more details.

## Scientific Basis

This system is inspired by:
- **Embodied Cognition Theory**: Cognition is grounded in bodily experiences
- **Affective Neuroscience**: Emotions and biology influence decision-making
- **Predictive Coding**: Brain as prediction machine modulated by physiology
- **Neurobiological Models**: Simplified models of real brain structures

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this system in your research, please cite:

```
@software{cognitive_system_2025,
  title={Cognitive System: Embodied Cognition AI},
  author={irfan0807},
  year={2025},
  url={https://github.com/irfan0807/cognitive-system}
}
```
