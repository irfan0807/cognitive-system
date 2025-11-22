# Cognitive System - Autonomous Embodied AI Simulation Framework

A comprehensive framework for creating autonomous characters based on **embodied cognition**, modeled after the Baby X system. This framework implements real-time neural network simulation, virtual biology, and experiential learning to create truly autonomous characters that can experience and learn from their environment.

## Overview

The Cognitive System is designed around the principle that **virtual biology drives virtual cognition and real-time learning and experience**. The entire system is centered on the character's virtual body, with all behaviors emerging from the interaction between virtual physiology, neural networks, and environmental experiences.

## Core Architecture

### I. Embodied Cognition Paradigm

The system implements the foundational principle of **embodied cognition**, where:

- The model is centered on its **virtual body**
- The character is animated live by **neural networks** utilizing a simplified brain simulation model
- The core goal is creating an **autonomous character that can actually experience things**
- The character is designed to **learn in real time** through experiences

### II. Virtual Biology and Nervous System

The character is entirely driven by its **virtual biology**, featuring:

#### Complete Virtual Nervous System
- **Whole virtual nervous system** controlling all aspects of character behavior
- **Virtual physiology** generating realistic physiological responses
- Real-time physiological adaptation (e.g., virtual heart and breathing rate increase when stressed)

#### Virtual Brain Model

The **virtual brain** is modeled live with specific functional structures:

1. **Brainstem**
   - Connects the brain to the body
   - Governs how the character acts, moves, and responds
   - Contains **oculomotor nuclei** for eye movement control
   - Manages autonomic functions (respiration, cardiovascular control)

2. **Chemical Regulation System**
   - **Pituitary gland** releasing simulated substances like **oxytocin**
   - Hormone regulation affecting emotional and physiological states
   - **Paraventricular nucleus** coordinating stress response and autonomic regulation

3. **Neural Network Controller**
   - Real-time processing of sensory and physiological inputs
   - Adaptive learning from experiences
   - Generation of motor commands and behavioral responses

### III. Behavior Generation and Memory System

#### Behavior Engine
- The complex virtual brain model **generates all the character's behaviors in real time**
- Motor behaviors, expressive behaviors, and attention allocation
- Environmental interaction capabilities (watching videos, playing games)
- **Automatic response to stimuli** like music (responding to the beat)

#### Multimodal Memory System

The system constructs a **history of memories** of events encountered through life. When a memory is formed (e.g., emotional conditioning by a word like "fire"), it is reconstructed **multimodally**:

- **Emotion at the time** (valence and arousal)
- **Visual stimuli at the time** (what was seen)
- **Auditory stimulation at the time** (what was heard)
- **Physiological state** (heart rate, hormone levels, stress)
- **Behavioral context** (what actions were being performed)

This multimodal integration enables rich experiential learning and realistic memory recall based on any combination of sensory, emotional, or contextual cues.

## Key Features

### 1. Real-Time Physiological Simulation
```python
# Virtual heart rate increases with stress
if character.stressed:
    virtual_heart.rate_increase()
    virtual_breathing.rate_increase()
```

### 2. Neural Network-Driven Behavior
All character behaviors emerge from neural network processing, not scripted responses. The character genuinely experiences and responds to its environment.

### 3. Experiential Learning
The character learns from its experiences in real-time, building associations between stimuli, emotions, and outcomes.

### 4. Environmental Interaction
- Watch videos and respond to visual content
- Play games on simulated screens
- Respond rhythmically to music
- Interact with objects in the environment

### 5. Chemical Regulation
Simulated hormones and neurotransmitters (oxytocin, cortisol, dopamine) regulate emotional and physiological states, creating realistic stress responses and social bonding behaviors.

## System Components

### Core Components
- `EmbodiedCognitionSystem`: Main system orchestrating all components
- `NeuralNetworkController`: Real-time neural processing and learning
- `VirtualNervousSystem`: Physiological simulation and regulation
- `VirtualBrain`: Brain structures (brainstem, pituitary, PVN, oculomotor nuclei)
- `MultimodalMemorySystem`: Experience storage and multimodal memory reconstruction
- `BehaviorEngine`: Real-time behavior generation

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                 Embodied Cognition System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │  Sensory Input   │────────▶│   Virtual Brain         │  │
│  │  - Visual        │         │  - Brainstem            │  │
│  │  - Auditory      │         │  - Oculomotor Nuclei    │  │
│  │  - Tactile       │         │  - Pituitary Gland      │  │
│  └──────────────────┘         │  - Paraventricular      │  │
│           │                    │    Nucleus              │  │
│           ▼                    └──────────┬──────────────┘  │
│  ┌──────────────────┐                     │                 │
│  │ Virtual Nervous  │◀────────────────────┘                 │
│  │    System        │                                       │
│  │  - Heart Rate    │         ┌─────────────────────────┐  │
│  │  - Breathing     │────────▶│  Behavior Engine        │  │
│  │  - Stress        │         │  - Motor Control        │  │
│  │  - Hormones      │         │  - Expressions          │  │
│  └──────────────────┘         │  - Attention            │  │
│           │                    └──────────┬──────────────┘  │
│           │                                │                 │
│           ▼                                ▼                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Neural Network Controller                     │  │
│  │  - Forward Processing                                 │  │
│  │  - Real-time Learning                                 │  │
│  │  - Adaptation                                         │  │
│  └─────────────────────────┬────────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Multimodal Memory System                         │  │
│  │  - Emotional Context                                  │  │
│  │  - Visual Memory                                      │  │
│  │  - Auditory Memory                                    │  │
│  │  - Physiological State                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Fundamental Relationship

**Virtual Biology → Virtual Cognition → Real-Time Learning & Experience**

The entire simulation ensures that virtual biology drives virtual cognition, which in turn drives real-time learning and experience. This creates a truly autonomous character that:

1. **Experiences** its environment through virtual senses and physiology
2. **Responds** authentically based on its current physiological and emotional state
3. **Learns** from experiences by forming multimodal memories
4. **Adapts** its behavior based on accumulated experiences
5. **Generates** behaviors in real-time through neural network processing

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

See the `examples/` directory for complete examples. Here's a basic usage:

```python
from cognitive_system import (
    EmbodiedCognitionSystem,
    VirtualNervousSystem,
    VirtualBrain,
    MultimodalMemorySystem,
    BehaviorEngine
)
from cognitive_system.core.neural_network import NeuralNetworkController

# Create system components
nervous_system = VirtualNervousSystem()
brain = VirtualBrain()
memory_system = MultimodalMemorySystem()
behavior_engine = BehaviorEngine()
neural_network = NeuralNetworkController(
    input_size=22,
    hidden_size=50,
    output_size=20
)

# Create main system
system = EmbodiedCognitionSystem()
system.setup(nervous_system, neural_network, memory_system, behavior_engine)

# Start simulation
system.start()

# Simulation loop
for t in range(1000):
    # Provide sensory input
    sensory_input = {
        'visual': np.random.randn(10) * 0.1,
        'auditory': np.random.randn(10) * 0.1,
    }
    
    # Update system
    output = system.update(delta_time=0.016, sensory_input=sensory_input)
    
    # Character learns from experiences
    if t % 100 == 0:
        system.learn_from_experience()
```

## Requirements

- Python 3.8+
- NumPy
- (Optional) TensorFlow/PyTorch for advanced neural network models

## Documentation

Detailed documentation is available in the `docs/` directory:

- `architecture.md`: Detailed system architecture
- `virtual_biology.md`: Virtual biology and nervous system details
- `memory_system.md`: Multimodal memory system guide
- `behavior_engine.md`: Behavior generation documentation
- `examples.md`: Usage examples and tutorials

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

MIT License - see `LICENSE` file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{cognitive_system,
  title={Cognitive System: Autonomous Embodied AI Simulation Framework},
  author={Cognitive System Team},
  year={2024},
  url={https://github.com/irfan0807/cognitive-system}
}
```

## Acknowledgments

This framework is inspired by the Baby X system and the principles of embodied cognition, implementing the requirement that virtual biology must drive virtual cognition and enable real-time learning through genuine experiences.