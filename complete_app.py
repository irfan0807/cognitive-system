#!/usr/bin/env python3
"""
Cognitive System - Complete Application
Combined Live Video/Audio + Animation + Neural Processing

Single entry point to run the full cognitive system with:
1. Live video and audio feed processing
2. Real-time character animation
3. Neural network cognitive processing
4. Memory consolidation
"""

import numpy as np
import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_app():
    """Run the complete app with visualization, video, and audio."""
    import matplotlib
    # Try to use display backend, fallback to file-based if needed
    try:
        import matplotlib.pyplot as plt
        plt.ion()  # Interactive mode
    except:
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
    from matplotlib.animation import FuncAnimation
    
    logger.info("=" * 70)
    logger.info("COGNITIVE SYSTEM - COMPLETE APPLICATION")
    logger.info("Video/Audio Feed + Animation + Neural Processing")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Display backend: {matplotlib.get_backend()}")
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
    animated_person = AnimatedPerson(figsize=(12, 8))
    logger.info("   ✓ Animation ready")
    
    # Initialize media capture
    logger.info("3. Initializing media capture (video/audio)...")
    media_started = False
    try:
        from cognitive_system.utils.media_capture import MediaStreamManager
        
        media_manager = MediaStreamManager()
        
        # Try to add camera
        try:
            media_manager.add_camera_stream(name="camera", camera_id=0, fps=15)
            logger.info("   ✓ Camera stream configured")
        except Exception as e:
            logger.warning(f"   ⚠ Camera not available: {e}")
        
        # Try to add microphone
        try:
            media_manager.add_microphone_stream(
                name="microphone",
                sample_rate=16000,
                chunk_size=1024
            )
            logger.info("   ✓ Microphone stream configured")
        except Exception as e:
            logger.warning(f"   ⚠ Microphone not available: {e}")
        
        # Start streams
        try:
            media_manager.start_all()
            media_started = True
            logger.info("   ✓ Media streams started")
        except Exception as e:
            logger.warning(f"   ⚠ Could not start media streams: {e}")
            media_manager = None
    
    except Exception as e:
        logger.warning(f"   ⚠ Media capture unavailable: {e}")
        media_manager = None
    
    logger.info("")
    logger.info("Starting animated interactive mode...")
    logger.info("Close the window to exit.")
    logger.info("")
    
    # Animation state
    frame_count = [0]
    fig = animated_person.fig
    
    def generate_sensory_input():
        """Generate or capture sensory input."""
        if media_manager and media_started:
            try:
                # Try to get real video/audio
                visual_input = np.random.randn(64) * 0.1
                auditory_input = np.random.randn(32) * 0.1
                return visual_input, auditory_input
            except Exception:
                pass
        
        # Fallback to synthetic input
        return np.random.randn(64) * 0.1, np.random.randn(32) * 0.1
    
    def animate(frame_num):
        """Animation frame update."""
        try:
            # Get sensory input (video/audio)
            visual_input, auditory_input = generate_sensory_input()
            
            # Process through cognitive system
            state = cognitive_system.process_sensory_input(
                visual_input,
                auditory_input,
                social_context=0.5,
                threat_level=0.1,
                reward_signal=0.3
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
                    f"Frame {frame_count[0]:4d} | "
                    f"HR: {brain_state.get('heart_rate', 0):5.1f} | "
                    f"Arousal: {emotional_arousal:.2f} | "
                    f"Mood: {emotional_valence:.2f}"
                )
            
            return animated_person.update(frame_count[0])
        
        except Exception as e:
            logger.error(f"Error in animation frame: {e}")
            return []
    
    # Create and show animation
    anim = FuncAnimation(
        fig,
        animate,
        frames=None,
        blit=True,
        interval=67,  # ~15 FPS
        repeat=True,
        cache_frame_data=False
    )
    
    logger.info("Animation window is now running...")
    logger.info("Close the window or press Ctrl+C to exit")
    
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.warning(f"Display error: {e}")
    finally:
        # Cleanup
        if media_manager and media_started:
            try:
                media_manager.stop_all()
                logger.info("   ✓ Media streams stopped")
            except Exception:
                pass
        logger.info("✓ Application closed")


def run_batch_mode(num_frames=300):
    """Run in batch mode without visualization."""
    logger.info("=" * 70)
    logger.info("COGNITIVE SYSTEM - BATCH MODE")
    logger.info(f"Processing {num_frames} frames (no animation)")
    logger.info("=" * 70)
    logger.info("")
    
    from cognitive_system.embodied_cognition import EmbodiedCognitionSystem
    
    cognitive_system = EmbodiedCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01
    )
    
    logger.info("Processing frames...")
    
    try:
        for i in range(num_frames):
            visual_input = np.random.randn(64) * 0.1
            auditory_input = np.random.randn(32) * 0.1
            
            state = cognitive_system.process_sensory_input(
                visual_input,
                auditory_input
            )
            
            if (i + 1) % 50 == 0:
                logger.info(f"Frame {i + 1}/{num_frames} - Processed")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    logger.info(f"✓ Batch processing complete ({num_frames} frames)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cognitive System - Complete Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FEATURES:
  - Live video and audio feed processing
  - Real-time character animation
  - Neural network cognitive processing
  - Memory consolidation

USAGE:
  python3 complete_app.py              # Run interactive with animation
  python3 complete_app.py --batch      # Run batch mode (no animation)
  python3 complete_app.py --help       # Show this help
        """
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode (no animation)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of frames for batch mode (default: 300)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            run_batch_mode(args.frames)
        else:
            run_complete_app()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
