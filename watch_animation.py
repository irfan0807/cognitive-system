#!/usr/bin/env python3
"""
Real-time terminal animation viewer
Shows the cognitive system animation live in the terminal with ASCII visualization
"""

import numpy as np
import sys
import logging
from pathlib import Path
import time

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def draw_person(arousal, stress, mood, heart_rate, attention):
    """Draw ASCII representation of animated person."""
    # Head color based on mood
    if mood > 0.5:
        head = "🟢"  # Happy - green
    elif mood < -0.5:
        head = "🔴"  # Sad - red
    else:
        head = "🟡"  # Neutral - yellow
    
    # Body size based on arousal
    body_size = "█" * max(1, int(arousal * 8))
    
    # Stress indicator
    stress_bar = "▓" * int(stress * 10) + "░" * (10 - int(stress * 10))
    
    # Heart pulse based on heart rate
    pulse = "❤" if int(time.time() * heart_rate / 60) % 2 else "♡"
    
    # Attention sparkles
    attention_sparkle = "✨" if attention > 0.7 else "·"
    
    art = f"""
    ╔════════════════════════════════════════════════╗
    ║                                                ║
    ║          {head} ANIMATED PERSON {head}              ║
    ║                                                ║
    ║         {body_size}                        ║
    ║                                                ║
    ║     Heart Rate:    {heart_rate:6.1f} bpm  {pulse}       ║
    ║     Arousal:       {arousal:.2f}              ║
    ║     Stress:  [{stress_bar}]     ║
    ║     Mood:          {mood:+.2f}              ║
    ║     Attention:     {attention:.2f}  {attention_sparkle}        ║
    ║                                                ║
    ╚════════════════════════════════════════════════╝
    """
    return art


def main():
    """Run real-time animation viewer."""
    logger.info("\n" + "=" * 50)
    logger.info("COGNITIVE SYSTEM - REAL-TIME ANIMATION VIEWER")
    logger.info("=" * 50 + "\n")
    
    # Initialize cognitive system
    logger.info("Initializing cognitive system...")
    from cognitive_system.embodied_cognition import EmbodiedCognitionSystem
    
    system = EmbodiedCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01
    )
    logger.info("✓ System ready\n")
    
    # Run animation frames
    num_frames = 100
    logger.info(f"Playing {num_frames} frames of animation...\n")
    
    try:
        for frame in range(num_frames):
            # Generate sensory input
            visual_input = np.random.randn(64) * 0.1
            auditory_input = np.random.randn(32) * 0.1
            
            # Add varying stimuli patterns
            social_context = 0.5 + 0.3 * np.sin(frame / 20)
            threat_level = 0.1 + 0.2 * np.sin(frame / 30)
            reward_signal = 0.3 + 0.2 * np.cos(frame / 25)
            
            # Process through cognitive system
            state = system.process_sensory_input(
                visual_input=visual_input,
                auditory_input=auditory_input,
                social_context=social_context,
                threat_level=threat_level,
                reward_signal=reward_signal
            )
            
            # Extract state
            emotional_valence, emotional_arousal = state.emotional_state
            brain_state = state.brain_state
            
            # Get values for display
            arousal = emotional_arousal
            stress = brain_state.get('stress_response', 0.0)
            mood = emotional_valence
            heart_rate = brain_state.get('heart_rate', 60.0)
            attention = brain_state.get('arousal', 0.5)
            
            # Clear screen (ANSI escape code)
            sys.stdout.write("\033[2J\033[H")
            
            # Draw ASCII animation
            logger.info(draw_person(arousal, stress, mood, heart_rate, attention))
            
            # Status bar
            progress = (frame + 1) / num_frames * 100
            bar_length = 40
            filled = int(bar_length * (frame + 1) / num_frames)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            logger.info(f"Progress: [{bar}] {progress:.0f}% ({frame + 1}/{num_frames})")
            logger.info(f"\nStimuli: Social={social_context:.2f} | Threat={threat_level:.2f} | Reward={reward_signal:.2f}")
            
            # Frame rate control (~5 FPS for terminal readability)
            time.sleep(0.2)
    
    except KeyboardInterrupt:
        logger.info("\n\n✓ Animation stopped by user")
    
    logger.info("\n" + "=" * 50)
    logger.info("Animation complete!")
    logger.info("=" * 50 + "\n")


if __name__ == '__main__':
    main()
