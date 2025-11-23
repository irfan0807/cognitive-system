#!/usr/bin/env python3
"""
Cognitive System - Complete App Launcher

Single entry point combining:
- Live video/audio feed processing  
- Real-time character animation
- Neural network processing

Run: python3 main_app.py
"""

import sys
import logging
from pathlib import Path

# Setup path to allow proper imports
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(
        description="Cognitive System - Complete Application",
        epilog="""
Examples:
  python3 main_app.py              # Run interactive with animation
  python3 main_app.py --batch      # Run batch mode (no animation)
        """
    )
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames")
    
    args = parser.parse_args()
    
    # Import core components
    from cognitive_system.embodied_cognition import EmbodiedCognitionSystem
    
    logger.info("=" * 70)
    logger.info("COGNITIVE SYSTEM - COMPLETE APPLICATION")
    logger.info("=" * 70)
    
    # Initialize components
    logger.info("Initializing system components...")
    cognitive_system = EmbodiedCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01
    )
    logger.info("✓ System initialized")
    
    if args.batch:
        # Batch mode
        logger.info(f"Running batch mode ({args.frames} frames)...")
        cognitive_system.start()
        
        try:
            for i in range(args.frames):
                # Process sensory input
                visual_input = np.random.randn(64) * 0.1
                auditory_input = np.random.randn(32) * 0.1
                
                output = cognitive_system.step(visual_input, auditory_input)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Frame {i + 1}/{args.frames} - Output shape: {len(output) if output else 0}")
        finally:
            cognitive_system.stop()
            logger.info("✓ Batch processing complete")
    
    else:
        # Interactive mode with animation
        try:
            from cognitive_system.visualization import AnimatedPerson
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            
            logger.info("Initializing animation...")
            animated_person = AnimatedPerson(figsize=(12, 8))
            logger.info("✓ Animation ready")
            logger.info("")
            logger.info("Running interactive mode...")
            logger.info("Close the window to exit.")
            logger.info("")
            
            cognitive_system.start()
            frame_count = [0]
            
            def animate(frame_num):
                try:
                    visual_input = np.random.randn(64) * 0.1
                    auditory_input = np.random.randn(32) * 0.1
                    
                    output = cognitive_system.step(visual_input, auditory_input)
                    
                    # Update animation based on output
                    animated_person.update_state(
                        arousal=0.5,
                        stress=0.0,
                        mood=0.5,
                        heart_rate=60.0,
                        attention=0.5
                    )
                    
                    frame_count[0] += 1
                    if frame_count[0] % 30 == 0:
                        logger.info(f"Frame {frame_count[0]}")
                    
                    return animated_person.artists
                except Exception as e:
                    logger.error(f"Animation error: {e}")
                    return []
            
            fig = animated_person.figure
            FuncAnimation(fig, animate, frames=None, blit=True, interval=67, repeat=True)
            
            plt.show()
        
        except ImportError as e:
            logger.error(f"Animation not available: {e}")
            logger.info("Running in text-only mode...")
            
            cognitive_system.start()
            logger.info("Processing frames... (press Ctrl+C to stop)")
            
            try:
                frame_count = 0
                while True:
                    sensory_input = {
                        'visual': np.random.randn(10) * 0.1,
                        'auditory': np.random.randn(10) * 0.1,
                    }
                    cognitive_system.update(1.0/15, sensory_input)
                    frame_count += 1
                    
                    if frame_count % 50 == 0:
                        state = nervous_system.get_state()
                        logger.info(
                            f"Frame {frame_count} | "
                            f"HR: {state.get('heart_rate', 0):.1f} | "
                            f"Stress: {state.get('stress_level', 0):.2f}"
                        )
            except KeyboardInterrupt:
                pass
        
        finally:
            cognitive_system.stop()
            logger.info("✓ App closed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
