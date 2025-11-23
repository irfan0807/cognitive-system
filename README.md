# Cognitive System - Embodied Cognition AI

An autonomous AI system with **embodied cognition**, driven by **neural networks** and a **simplified brain simulation model** for **real-time learning**. The system includes a **complete virtual nervous system** and **physiology** where **virtual biology drives virtual cognition**.

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

## Running the Demos

### Basic Cognitive System Demo

```bash
python examples/demo.py
```

This will run a demonstration showing:
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
