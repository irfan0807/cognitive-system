#!/usr/bin/env python3
"""
Save animated person visualization to video file
"""

import numpy as np
import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')  # Use file-based backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_animation_demo():
    """Save animated visualization to GIF file."""
    logger.info("=" * 70)
    logger.info("Cognitive System Animation - Saving to File")
    logger.info("=" * 70)
    logger.info("")
    
    # Initialize cognitive system
    logger.info("1. Initializing cognitive system...")
    from cognitive_system.embodied_cognition import EmbodiedCognitionSystem
    cognitive_system = EmbodiedCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01
    )
    logger.info("   ✓ Cognitive system ready")
    
    # Initialize animation
    logger.info("2. Initializing animation visualization...")
    from src.cognitive_system.visualization import AnimatedPerson
    animated_person = AnimatedPerson(figsize=(14, 8))
    logger.info("   ✓ Animation ready")
    
    # Animation state
    frame_count = [0]
    max_frames = 300
    fig = animated_person.fig
    
    def animate(frame_num):
        """Animation frame update."""
        try:
            # Generate sensory input (video/audio simulation)
            visual_input = np.random.randn(64) * 0.1
            auditory_input = np.random.randn(32) * 0.1
            
            # Process through cognitive system
            state = cognitive_system.process_sensory_input(
                visual_input,
                auditory_input,
                social_context=0.5 + 0.3 * np.sin(frame_count[0] / 50),
                threat_level=0.1 + 0.2 * np.sin(frame_count[0] / 100),
                reward_signal=0.3 + 0.2 * np.cos(frame_count[0] / 75)
            )
            
            # Extract emotion and physiology
            emotional_valence, emotional_arousal = state.emotional_state
            brain_state = state.brain_state
            
            # Update animation with state dictionary
            animated_person.update_state({
                'arousal': emotional_arousal,
                'stress': brain_state.get('stress_response', 0.0),
                'mood': emotional_valence,
                'heart_rate': brain_state.get('heart_rate', 60.0),
                'attention_level': brain_state.get('arousal', 0.5),
                'emotional_valence': emotional_valence,
                'emotional_arousal': emotional_arousal,
                'oxytocin': brain_state.get('oxytocin', 0.5)
            })
            
            frame_count[0] += 1
            
            # Print status periodically
            if frame_count[0] % 30 == 0:
                logger.info(
                    f"Frame {frame_count[0]:3d}/{max_frames} | "
                    f"HR: {brain_state.get('heart_rate', 0):5.1f} | "
                    f"Arousal: {emotional_arousal:.2f} | "
                    f"Mood: {emotional_valence:.2f}"
                )
            
            return animated_person.update(frame_count[0])
        
        except Exception as e:
            logger.error(f"Error in animation frame: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    logger.info("")
    logger.info(f"3. Creating animation ({max_frames} frames)...")
    
    # Create animation
    anim = FuncAnimation(
        fig,
        animate,
        frames=max_frames,
        blit=True,
        interval=67,  # ~15 FPS
        repeat=False,
        cache_frame_data=False
    )
    
    # Save as GIF
    output_file = "cognitive_system_animation.gif"
    logger.info(f"   Saving to {output_file}...")
    
    try:
        writer = PillowWriter(fps=15)
        anim.save(output_file, writer=writer, dpi=80)
        logger.info(f"   ✓ Animation saved: {output_file}")
        logger.info(f"   File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"   ✗ Failed to save animation: {e}")
        logger.info("   Trying to save frames as PNG images instead...")
        
        # Fallback: save individual frames
        output_dir = Path("animation_frames")
        output_dir.mkdir(exist_ok=True)
        
        frame_count[0] = 0
        for i in range(10):  # Save first 10 frames as example
            visual_input = np.random.randn(64) * 0.1
            auditory_input = np.random.randn(32) * 0.1
            
            state = cognitive_system.process_sensory_input(
                visual_input,
                auditory_input,
                social_context=0.5,
                threat_level=0.1,
                reward_signal=0.3
            )
            
            emotional_valence, emotional_arousal = state.emotional_state
            brain_state = state.brain_state
            
            animated_person.update_state({
                'arousal': emotional_arousal,
                'stress': brain_state.get('stress_response', 0.0),
                'mood': emotional_valence,
                'heart_rate': brain_state.get('heart_rate', 60.0),
                'attention_level': brain_state.get('arousal', 0.5),
                'emotional_valence': emotional_valence,
                'emotional_arousal': emotional_arousal,
                'oxytocin': brain_state.get('oxytocin', 0.5)
            })
            
            animated_person.update(i)
            frame_path = output_dir / f"frame_{i:03d}.png"
            plt.savefig(frame_path, dpi=80, bbox_inches='tight')
            logger.info(f"   Saved: {frame_path}")
        
        logger.info(f"   ✓ Frames saved to {output_dir}/")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Animation demo complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    save_animation_demo()
