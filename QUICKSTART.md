# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/irfan0807/cognitive-system.git
cd cognitive-system

# Install dependencies
pip install -r requirements.txt

# Install package (optional)
pip install -e .
```

## Basic Usage

```python
from cognitive_system import (
    EmbodiedCognitionSystem,
    VirtualNervousSystem,
    VirtualBrain,
    MultimodalMemorySystem,
    BehaviorEngine
)
from cognitive_system.core.neural_network import NeuralNetworkController
import numpy as np

# Initialize components
nervous_system = VirtualNervousSystem()
brain = VirtualBrain()
memory = MultimodalMemorySystem()
behavior = BehaviorEngine()
neural_net = NeuralNetworkController(input_size=22, hidden_size=50, output_size=20)

# Create system
system = EmbodiedCognitionSystem()
system.setup(nervous_system, neural_net, memory, behavior)
system.start()

# Simulation loop
for step in range(100):
    sensory_input = {
        'visual': np.random.randn(10) * 0.1,
        'auditory': np.random.randn(10) * 0.1,
    }
    output = system.update(delta_time=0.016, sensory_input=sensory_input)
    
    # Access results
    print(f"Heart rate: {output['physiological_state']['heart_rate']:.1f} bpm")
    print(f"Stress: {output['physiological_state']['stress_level']:.2f}")

system.stop()
```

## Running Examples

```bash
# Basic simulation
PYTHONPATH=. python examples/basic_example.py

# Environmental interaction (music, video, games)
PYTHONPATH=. python examples/environmental_interaction.py

# Emotional conditioning demonstration
PYTHONPATH=. python examples/emotional_conditioning.py
```

## Running Tests

```bash
python tests/test_basic.py
```

## Key Components

### 1. Virtual Nervous System
- Manages physiological state (heart rate, breathing, stress)
- Chemical regulation (oxytocin, cortisol, dopamine)
- Responds to stimuli (music beats, stress)

### 2. Virtual Brain
- **Brainstem:** Motor control, autonomic functions
- **Pituitary Gland:** Hormone release (oxytocin)
- **Paraventricular Nucleus:** Stress response
- **Oculomotor Nuclei:** Eye movements

### 3. Memory System
- Multimodal memory (visual + auditory + emotional + physiological)
- Experience storage and recall
- Memory consolidation and decay

### 4. Behavior Engine
- Real-time behavior generation
- Motor, expressive, attention behaviors
- Environmental interaction (videos, games, music)

## Common Operations

### Set Physiological State
```python
# For testing/simulation
nervous_system.simulate_emotional_response(
    valence=-0.5,  # Negative emotion
    arousal=0.8,   # High arousal
    stress=0.7     # High stress
)
```

### Respond to Music
```python
nervous_system.respond_to_music(beat_strength=1.0, delta_time=0.016)
```

### Recall Memories
```python
cue = {
    'emotional_state': {'valence': -0.5, 'arousal': 0.8}
}
memories = memory_system.recall_memories(cue, top_k=5)
```

### Interact with Environment
```python
result = behavior_engine.interact_with_environment(
    'music',
    {'beat_strength': 1.0, 'beat_frequency': 2.0}
)
```

## Configuration

See `config/default_config.yaml` for all configuration options.

## Documentation

- `README.md` - Overview and features
- `docs/architecture.md` - Detailed architecture
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

## Requirements Mapping

| Requirement | Implementation |
|-------------|----------------|
| Embodied cognition | `EmbodiedCognitionSystem` |
| Virtual nervous system | `VirtualNervousSystem` |
| Brain structures | `VirtualBrain` with brainstem, pituitary, PVN |
| Multimodal memory | `MultimodalMemorySystem` |
| Real-time behavior | `BehaviorEngine` |
| Environmental interaction | `interact_with_environment()` |
| Learning | `learn_from_experience()` |

## Support

For issues or questions, please refer to the documentation or create an issue on GitHub.
