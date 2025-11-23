# Stress Control System Summary

## How to Increase/Decrease Animation Stress

### Method 1: Using Stress Controller Directly

```python
from cognitive_system.utils.stress_controller import StressController

# Create controller
stress_controller = StressController(initial_stress=0.4, step_size=0.1)

# INCREASE stress
stress_controller.increase_stress()        # Increase by 0.1
stress_controller.increase_stress(0.2)     # Increase by 0.2

# DECREASE stress
stress_controller.decrease_stress()        # Decrease by 0.1
stress_controller.decrease_stress(0.15)    # Decrease by 0.15

# SET to specific level (0.0 to 1.0)
stress_controller.set_stress(0.7)          # Set to 70%
stress_controller.set_stress(0.0)          # Set to 0% (very relaxed)
stress_controller.set_stress(1.0)          # Set to 100% (extremely stressed)
```

### Method 2: Using Preset Levels

```python
from cognitive_system.utils.stress_controller import StressLevel

# Quick access to common stress levels
stress_controller.set_preset(StressLevel.VERY_LOW)    # 0%
stress_controller.set_preset(StressLevel.LOW)         # 20%
stress_controller.set_preset(StressLevel.NORMAL)      # 40% (default)
stress_controller.set_preset(StressLevel.MODERATE)    # 60%
stress_controller.set_preset(StressLevel.HIGH)        # 80%
stress_controller.set_preset(StressLevel.VERY_HIGH)   # 100%
```

### Method 3: Keyboard Commands (Interactive)

```bash
# Run interactive demo
python3 stress_demo.py --interactive

# Then use keyboard:
# '+' or '=' : Increase stress
# '-' or '_' : Decrease stress
# '0' or '1': Very Low (0%)
# '2'       : Low (20%)
# '3'       : Normal (40%)
# '4'       : Moderate (60%)
# '5'       : High (80%)
# '6'       : Very High (100%)
# 'r'       : Reset to normal (40%)
```

---

## What Changes When Stress Increases/Decreases

### Visual Changes in Animation

When stress **INCREASES**:
- 👁️ Eyes become wider (higher arousal)
- ❤️ Breathing becomes faster
- 😰 Facial expression becomes more tense
- 💪 Body posture becomes more rigid
- 📈 Heart rate increases (shown in metrics)

When stress **DECREASES**:
- 😌 Eyes relax
- 🌬️ Breathing normalizes
- 😊 Facial expression becomes more calm
- 🧘 Body relaxes
- 📉 Heart rate decreases

### Parameter Changes

| Parameter | Low Stress (0.2) | Normal (0.4) | High Stress (0.8) |
|-----------|------------------|--------------|-------------------|
| Arousal | 0.44 | 0.56 | 0.86 |
| Heart Rate | 68 bpm | 76 bpm | 92 bpm |
| Mood | 0.74 | 0.58 | 0.46 |
| Attention | 0.54 | 0.55 | 0.66 |

---

## Quick Examples

### Example 1: Gradually Increase Stress

```python
from cognitive_system.utils.stress_controller import StressController

controller = StressController(initial_stress=0.4, step_size=0.1)

# Gradually increase stress
for i in range(5):
    controller.increase_stress()
    print(controller.get_status_string())

# Output:
# [██████████░░░░░░░░░░] 50.0% - Normal
# [███████████░░░░░░░░░░] 60.0% - Moderate
# [████████████░░░░░░░░░░] 70.0% - Tense
# [█████████████░░░░░░░░░░] 80.0% - High
# [██████████████░░░░░░░░░░] 90.0% - Very Tense
```

### Example 2: Stress Scenario

```python
from cognitive_system.utils.stress_controller import StressLevel

controller = StressController()

# Start: Person is relaxed
controller.set_preset(StressLevel.NORMAL)
print("📍 Normal day:", controller.get_status_string())

# Event: Pressure builds
controller.increase_stress(0.15)
print("⚡ Under pressure:", controller.get_status_string())

# Crisis: Emergency
controller.set_preset(StressLevel.VERY_HIGH)
print("🚨 Emergency:", controller.get_status_string())

# Resolution: Stress relief
controller.set_preset(StressLevel.LOW)
print("✅ Relaxing:", controller.get_status_string())
```

### Example 3: Direct Numeric Control

```python
# Set exact stress values
controller.set_stress(0.0)      # 0% - Completely relaxed
controller.set_stress(0.25)     # 25% - Slightly stressed
controller.set_stress(0.5)      # 50% - Moderately stressed
controller.set_stress(0.75)     # 75% - Very stressed
controller.set_stress(1.0)      # 100% - Extremely stressed
```

---

## Run Stress Demos

```bash
# See all demos with animations
python3 stress_demo.py

# Run specific demo
python3 stress_demo.py --basic       # Increase/decrease demo
python3 stress_demo.py --presets     # Show preset levels
python3 stress_demo.py --bridge      # Animation parameter changes
python3 stress_demo.py --interactive # Interactive keyboard control
```

---

## Integration with Main App

To use stress control with the animation app:

1. **Create stress controller**:
```python
from cognitive_system.utils.stress_controller import StressController, StressAnimationBridge

stress_controller = StressController(initial_stress=0.4)
bridge = StressAnimationBridge(stress_controller)
```

2. **Apply stress to animation**:
```python
# In your animation loop:
animation_state = bridge.get_animation_state_with_stress(base_state)
animated_person.update_state(animation_state)
```

3. **Control stress dynamically**:
```python
# During execution:
stress_controller.increase_stress()  # Make character more stressed
stress_controller.decrease_stress()  # Make character relax
```

---

## File Locations

- **Stress Controller**: `cognitive_system/utils/stress_controller.py`
- **Demo App**: `stress_demo.py`
- **Guide**: `STRESS_CONTROL_GUIDE.md`

---

## Key Features

✅ **Simple API**: Easy increase/decrease operations  
✅ **Presets**: Quick access to common stress levels  
✅ **Callbacks**: React to stress changes  
✅ **Interactive**: Keyboard control support  
✅ **Animation Bridge**: Automatic animation parameter adjustment  
✅ **Visualizations**: ASCII progress bars and descriptions  
✅ **Flexible**: Use programmatically or interactively  

---

## Next Steps

1. Run the demo: `python3 stress_demo.py`
2. Try interactive mode: `python3 stress_demo.py --interactive`
3. Integrate with your app using the examples above
4. Read `STRESS_CONTROL_GUIDE.md` for complete documentation

Enjoy controlling your character's stress levels! 🎮
