#!/usr/bin/env python3
"""
Quick test to verify animation displays
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import logging
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_animation():
    """Test that animation window shows."""
    logger.info("Creating test animation...")
    
    # Import components
    from cognitive_system.visualization import AnimatedPerson
    
    # Create animated person with video support
    animated_person = AnimatedPerson(figsize=(15, 8), with_video=False)
    logger.info("✓ AnimatedPerson created")
    
    # Frame counter
    frame_count = [0]
    
    def animate(frame_num):
        """Simple animation update."""
        frame_count[0] += 1
        
        # Update with simple state
        animated_person.update_state({
            'arousal': 0.5 + 0.3 * np.sin(frame_count[0] / 20),
            'stress': 0.3 * np.sin(frame_count[0] / 30),
            'mood': 0.5 + 0.2 * np.cos(frame_count[0] / 40),
            'heart_rate': 60 + 10 * np.sin(frame_count[0] / 25),
            'attention': 0.5,
            'emotional_valence': 0.1 * np.sin(frame_count[0] / 35),
            'emotional_arousal': 0.5,
            'oxytocin': 0.5
        })
        
        # Get animation elements
        artists = animated_person.update()
        
        if frame_count[0] % 20 == 0:
            logger.info(f"Frame: {frame_count[0]}")
        
        return artists
    
    # Create animation
    fig = animated_person.fig
    logger.info("Creating FuncAnimation...")
    anim = FuncAnimation(fig, animate, frames=None, blit=False, interval=67, repeat=True)
    logger.info("✓ Animation created")
    
    logger.info("Showing window... (close to exit)")
    plt.show()
    logger.info("✓ Test complete")

if __name__ == "__main__":
    try:
        test_animation()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
