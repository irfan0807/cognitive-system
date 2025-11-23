# Stress Control Guide

## Overview

The cognitive system now includes a **Stress Controller** that allows you to dynamically increase, decrease, and control the stress level of the animated character in real-time.

## Quick Start

### 1. Run Stress Control Demos

```bash
# See all demos
python3 stress_demo.py

# Run specific demos
python3 stress_demo.py --basic      # Basic increase/decrease demo
python3 stress_demo.py --presets    # Preset levels demo
python3 stress_demo.py --bridge     # Animation bridge demo
python3 stress_demo.py --interactive # Interactive command line
```

### 2. Run App with Stress Control

```bash
python3 app_with_video.py --test
```

During runtime, you can control stress by sending commands programmatically.

---

## Using Stress Controller Programmatically

### Basic Usage

```python
from cognitive_system.utils.stress_controller import StressController

# Create controller
controller = StressController(initial_stress=0.4, step_size=0.1)

# Increase stress
controller.increase_stress()           # Increase by step_size (0.1)
controller.increase_stress(0.2)        # Increase by custom amount

# Decrease stress
controller.decrease_stress()           # Decrease by step_size
controller.decrease_stress(0.15)       # Decrease by custom amount

# Set exact level
controller.set_stress(0.7)             # Set to 70% stress

# Get current stress
stress = controller.get_stress()       # Returns: 0.7
description = controller.get_stress_description()  # Returns: "Very Tense"
status = controller.get_status_string()  # Returns: "[████████████████░░] 70.0% - Very Tense"
```

### Preset Stress Levels

```python
from cognitive_system.utils.stress_controller import StressController, StressLevel

controller = StressController()

# Use presets
controller.set_preset(StressLevel.VERY_LOW)    # 0.0%
controller.set_preset(StressLevel.LOW)         # 20.0%
controller.set_preset(StressLevel.NORMAL)      # 40.0% (default)
controller.set_preset(StressLevel.MODERATE)    # 60.0%
controller.set_preset(StressLevel.HIGH)        # 80.0%
controller.set_preset(StressLevel.VERY_HIGH)   # 100.0%
```

### Callbacks

```python
controller = StressController()

# Define callback function
def on_stress_change(new_stress):
    print(f"Stress changed to: {new_stress:.2f}")

# Register callback
controller.on_stress_change = on_stress_change

# Now any stress change triggers the callback
controller.increase_stress()  # Prints: "Stress changed to: 0.50"
```

---

## Animation Bridge

The `StressAnimationBridge` automatically adjusts animation parameters based on stress level:

```python
from cognitive_system.utils.stress_controller import StressController, StressAnimationBridge

controller = StressController()
bridge = StressAnimationBridge(controller)

# Base animation state
state = {
    'arousal': 0.5,
    'stress': 0.4,
    'mood': 0.7,
    'heart_rate': 60.0,
    'attention': 0.5,
}

# Modify state based on stress
controller.set_stress(0.8)  # High stress
modified_state = bridge.get_animation_state_with_stress(state)

# Results:
# - Stress: 0.80
# - Arousal: increased to 0.86 (0.3 + 0.7 * 0.8)
# - Heart Rate: increased to 92 bpm (60 + 40 * 0.8)
# - Mood: decreased to 0.46 (0.7 - 0.3 * 0.8)
# - Attention: increased to 0.66 (0.5 + 0.2 * 0.8)
```

### How Stress Affects Animation

| Property | Effect | Formula |
|----------|--------|---------|
| **Arousal** | Increases with stress | `0.3 + 0.7 * stress` |
| **Heart Rate** | Increases with stress | `60 + 40 * stress` |
| **Mood** | Decreases with stress | `mood - 0.3 * stress` |
| **Attention** | Increases with stress | `min(1.0, attention + 0.2 * stress)` |

---

## Interactive Keyboard Control

Use `InteractiveStressController` for keyboard input:

```python
from cognitive_system.utils.stress_controller import InteractiveStressController

controller = InteractiveStressController(initial_stress=0.4, step_size=0.1)

# Handle keyboard input
key_pressed = '+'
new_stress = controller.handle_key_press(key_pressed)  # Increase
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `+` or `=` | Increase stress by step |
| `-` or `_` | Decrease stress by step |
| `0` or `1` | Set to Very Low (0%) |
| `2` | Set to Low (20%) |
| `3` | Set to Normal (40%) |
| `4` | Set to Moderate (60%) |
| `5` | Set to High (80%) |
| `6` | Set to Very High (100%) |
| `R` | Reset to Normal (40%) |

---

## Integration with Main App

The main app (`app_with_video.py`) can be extended to use stress control:

```python
from cognitive_system.utils.stress_controller import StressAnimationBridge

# In your animation loop:

# Create bridge
bridge = StressAnimationBridge(stress_controller)

# Get animation state with stress applied
animation_state = bridge.get_animation_state_with_stress(base_state)

# Update animation
animated_person.update_state(animation_state)
```

---

## Stress Descriptions

The system provides human-readable stress descriptions:

```python
controller.set_stress(0.1)   # "Very Relaxed"
controller.set_stress(0.3)   # "Relaxed"
controller.set_stress(0.5)   # "Normal"
controller.set_stress(0.7)   # "Tense"
controller.set_stress(0.9)   # "Very Tense"
controller.set_stress(1.0)   # "Extremely Stressed"
```

---

## Stress Visualization

Visual stress bar (ASCII):

```
[░░░░░░░░░░░░░░░░░░░░] 0.0% - Very Relaxed
[████░░░░░░░░░░░░░░░░] 20.0% - Relaxed
[████████░░░░░░░░░░░░] 40.0% - Normal
[███████████░░░░░░░░░░] 55.0% - Tense
[███████████████░░░░░░] 75.0% - Very Tense
[█████████████████████] 100.0% - Extremely Stressed
```

---

## Examples

### Example 1: Simple Stress Increase

```python
from cognitive_system.utils.stress_controller import StressController

controller = StressController(initial_stress=0.4)

print(f"Initial: {controller.get_status_string()}")
# Output: [████████░░░░░░░░░░░░] 40.0% - Normal

controller.increase_stress()
print(f"After increase: {controller.get_status_string()}")
# Output: [██████████░░░░░░░░░░] 50.0% - Normal (tending toward tense)
```

### Example 2: Stress Response Scenario

```python
from cognitive_system.utils.stress_controller import StressController, StressLevel

controller = StressController()

# Scenario: Person receives good news
controller.set_preset(StressLevel.VERY_LOW)
print(f"Relaxed: {controller.get_status_string()}")

# Scenario: Pressure builds up
controller.increase_stress(0.1)
controller.increase_stress(0.1)
print(f"Under pressure: {controller.get_status_string()}")

# Scenario: Crisis moment
controller.set_preset(StressLevel.VERY_HIGH)
print(f"Emergency: {controller.get_status_string()}")

# Scenario: Crisis resolved
controller.set_preset(StressLevel.LOW)
print(f"Recovering: {controller.get_status_string()}")
```

### Example 3: Animation Response to Stress

```python
from cognitive_system.utils.stress_controller import StressController, StressAnimationBridge

controller = StressController(initial_stress=0.4)
bridge = StressAnimationBridge(controller)

base_state = {
    'arousal': 0.5,
    'stress': 0.4,
    'mood': 0.8,
    'heart_rate': 70.0,
    'attention': 0.5,
}

# Low stress scenario
controller.set_stress(0.2)
state_low = bridge.get_animation_state_with_stress(base_state)
print(f"Low stress - HR: {state_low['heart_rate']:.0f}, Mood: {state_low['mood']:.2f}")
# Output: Low stress - HR: 68, Mood: 0.74

# High stress scenario
controller.set_stress(0.8)
state_high = bridge.get_animation_state_with_stress(base_state)
print(f"High stress - HR: {state_high['heart_rate']:.0f}, Mood: {state_high['mood']:.2f}")
# Output: High stress - HR: 92, Mood: 0.56
```

---

## Testing

Run the demo to see all stress control features:

```bash
# Interactive demo
python3 stress_demo.py --interactive

# You can then:
# - Type '+' to increase
# - Type '-' to decrease
# - Type numbers 0-6 for presets
# - Type 'r' to reset
# - Type a float like '0.75' to set exact level
# - Type 'quit' to exit
```

---

## Architecture

```
StressController (Base)
├── Manages stress value (0.0 - 1.0)
├── Provides increase/decrease/set methods
├── Triggers callbacks on change
└── Generates descriptions

InteractiveStressController (Extended)
├── Inherits from StressController
└── Adds keyboard input handling

StressAnimationBridge
├── Uses StressController
└── Maps stress to animation parameters
```

---

## Summary

- **StressController**: Basic stress management
- **InteractiveStressController**: Keyboard input support
- **StressAnimationBridge**: Connects stress to animation parameters
- **Presets**: Quick access to common stress levels
- **Callbacks**: Reactive programming support
- **Descriptions**: Human-readable status

Now you can easily control and visualize stress levels in your cognitive system animation!
