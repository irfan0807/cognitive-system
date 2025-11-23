# Animated Person Visualization

This module provides an animated visual representation of a person/character that is connected to the cognitive system's neural network. The character's appearance and movements respond in real-time to the neural network's state.

## Features

- **Character Visualization**: Simplified stick figure with head, body, arms, and legs
- **Emotional Representation**: Head color changes based on emotional valence (green for positive, red for negative)
- **Physiological Animation**: 
  - Breathing-like movement based on arousal
  - Heartbeat pulsing visualization
  - Arm and leg movements driven by neural network state
- **Real-time Metrics Display**: Shows neural network state including:
  - Heart Rate (BPM)
  - Arousal Level
  - Stress Level
  - Mood (Serotonin)
  - Attention Level
  - Oxytocin (Social Bonding)
  - Emotional Valence
  - Emotional Arousal

## Usage

### Basic Usage

```python
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem
from cognitive_system.visualization import AnimatedPerson
import numpy as np

# Initialize cognitive system
system = EmbodiedCognitionSystem(
    visual_dim=64,
    auditory_dim=32,
    attention_dim=128,
    learning_rate=0.01
)

# Create animated person
person = AnimatedPerson(figsize=(14, 8))

# Process sensory input through neural network
visual_input = np.random.randn(64) * 0.3
auditory_input = np.random.randn(32) * 0.3

state = system.process_sensory_input(
    visual_input=visual_input,
    auditory_input=auditory_input,
    social_context=0.8,
    threat_level=0.0,
    reward_signal=0.7
)

# Update animated person with neural network state
summary = system.get_state_summary()
person.update_state(summary)
person.update(frame=0)

# Display
person.show()
```

### Running Examples

```bash
# Simple demo (generates static images)
cd examples
python animated_person_simple_demo.py

# Full animated demo (requires display)
cd examples
python animated_person_demo.py
```

## How It Works

The `AnimatedPerson` class receives the cognitive system's state and translates it into visual representations:

1. **Emotional Color Mapping**: 
   - Positive valence → Greenish colors
   - Negative valence → Reddish colors
   - Mood influences color saturation

2. **Movement Generation**:
   - Arousal level controls breathing amplitude and limb movement
   - Stress level affects arm positioning (higher stress = arms closer to body)
   - Heart rate drives the pulsing heartbeat indicator

3. **Metrics Visualization**:
   - Each neural network parameter is displayed as a horizontal bar
   - Colors change based on values (e.g., red for high stress, green for positive mood)

## Neural Network Connection

The animated person is directly connected to these neural network outputs:

- `arousal`: From brain stem (controls general activation)
- `stress`: From paraventricular nucleus (stress response)
- `mood`: Serotonin levels from nervous system
- `heart_rate`: From autonomic nervous system
- `emotional_valence`: Positive/negative emotion (-1 to 1)
- `emotional_arousal`: Intensity of emotion
- `attention_level`: Norepinephrine-driven attention
- `oxytocin`: Social bonding hormone from pituitary gland

## Output Examples

The visualization shows different states:

- **Calm & Happy**: Green head, relaxed posture, low stress bars
- **Stressed & Anxious**: Red head, tense posture, high stress/arousal bars
- **Social Support**: Balanced state with high oxytocin despite some stress

## Requirements

- matplotlib >= 3.5.0
- numpy >= 1.21.0
- scipy >= 1.7.0
