#!/usr/bin/env python3
"""Test script to verify animation window works with video feed."""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cognitive_system.visualization import AnimatedPerson

print("Creating animated person...")
animated_person = AnimatedPerson(figsize=(12, 8))
print("✓ Animation created")

# Simulate state changes
frame_count = [0]

def animate(frame_num):
    """Update animation frame."""
    # Simulate changing state
    arousal = 0.5 + 0.3 * np.sin(frame_count[0] / 30)
    stress = 0.3 * np.sin(frame_count[0] / 50)
    mood = 0.5 + 0.2 * np.cos(frame_count[0] / 40)
    
    animated_person.update_state(
        arousal=arousal,
        stress=stress,
        mood=mood,
        heart_rate=60 + 10 * np.sin(frame_count[0] / 20),
        attention=0.5 + 0.2 * np.cos(frame_count[0] / 60),
        emotional_valence=0.3 * np.sin(frame_count[0] / 35),
        emotional_arousal=0.5 + 0.2 * np.sin(frame_count[0] / 45),
        oxytocin=0.5 + 0.1 * np.sin(frame_count[0] / 100)
    )
    
    frame_count[0] += 1
    
    if frame_count[0] % 30 == 0:
        print(f"Frame {frame_count[0]}")
    
    return animated_person.artists

print("Setting up animation...")
fig = animated_person.figure
animation = FuncAnimation(fig, animate, frames=None, blit=True, interval=67, repeat=True)
print("✓ Animation ready. Close window to exit.")
print("")

plt.show()
print("Animation closed.")
