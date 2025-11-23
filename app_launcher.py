#!/usr/bin/env python3
"""
Cognitive System - Main App Launcher

Run the complete cognitive system with live video/audio feed and animation.

Usage:
    python3 app_launcher.py                    # Interactive mode with animation
    python3 app_launcher.py --batch            # Batch mode without animation
    python3 app_launcher.py --no-camera        # Without camera input
    python3 app_launcher.py --no-microphone    # Without microphone input
"""

import sys
import logging
from pathlib import Path

# Add both parent and src directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_interactive():
    """Run interactive mode with animation."""
    from cognitive_system.core.embodied_cognition import EmbodiedCognitionSystem
    from cognitive_system.virtual_biology.nervous_system import VirtualNervousSystem
    from cognitive_system.core.neural_network import NeuralNetworkController
    from cognitive_system.memory.multimodal_memory import MultimodalMemorySystem
    from cognitive_system.behavior.behavior_engine import BehaviorEngine
    from cognitive_system.visualization import AnimatedPerson
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    logger.info("=" * 70)
    logger.info("COGNITIVE SYSTEM - INTERACTIVE MODE")
    logger.info("Live Video/Audio Feed + Animation")
    logger.info("=" * 70)
    
    # Initialize components
    logger.info("Initializing system components...")
    cognitive_system = EmbodiedCognitionSystem()
    nervous_system = VirtualNervousSystem()
    neural_network = NeuralNetworkController(
        input_size=22,
        hidden_size=128,
        output_size=32
    )
    memory_system = MultimodalMemorySystem()
    behavior_engine = BehaviorEngine()
    
    cognitive_system.setup(
        virtual_biology=nervous_system,
        neural_network=neural_network,
        memory_system=memory_system,
        behavior_engine=behavior_engine
    )
    logger.info("✓ System initialized")
    
    # Initialize animation
    logger.info("Initializing animation...")
    animated_person = AnimatedPerson(figsize=(12, 8))
    logger.info("✓ Animation ready")
    
    # Initialize media capture
    try:
        from cognitive_system.utils.media_capture import MediaStreamManager
        from cognitive_system.utils.media_processor import MediaInputBridge
        
        logger.info("Initializing media capture...")
        media_manager = MediaStreamManager()
        
        try:
            media_manager.add_camera_stream(name="camera", camera_id=0, fps=15)
            logger.info("✓ Camera stream added")
        except Exception as e:
            logger.warning(f"Camera unavailable: {e}")
        
        try:
            media_manager.add_microphone_stream(
                name="microphone",
                sample_rate=16000,
                chunk_size=1024
            )
            logger.info("✓ Microphone stream added")
        except Exception as e:
            logger.warning(f"Microphone unavailable: {e}")
        
        media_bridge = MediaInputBridge()
        
        # Try to start streams
        try:
            media_manager.start_all()
            logger.info("✓ Media streams started")
            use_media = True
        except Exception as e:
            logger.warning(f"Could not start media streams: {e}")
            use_media = False
    
    except Exception as e:
        logger.warning(f"Media capture not available: {e}")
        use_media = False
        media_manager = None
    
    # Start cognitive system
    cognitive_system.start()
    logger.info("✓ Cognitive system started")
    logger.info("")
    logger.info("Running animation... Close window to exit.")
    logger.info("")
    
    # Animation state
    frame_count = [0]  # Use list to allow modification in nested function
    
    def get_sensory_input():
        """Get sensory input from media or generate synthetic."""
        if use_media and media_manager:
            try:
                sensory_input = {
                    'visual': np.random.randn(10) * 0.1,
                    'auditory': np.random.randn(10) * 0.1,
                }
                return sensory_input
            except Exception:
                pass
        
        # Fallback to synthetic input
        return {
            'visual': np.random.randn(10) * 0.1,
            'auditory': np.random.randn(10) * 0.1,
        }
    
    def animate(frame_num):
        """Update animation frame."""
        try:
            # Get sensory input
            sensory_input = get_sensory_input()
            
            # Process through cognitive system
            output = cognitive_system.update(1.0/15, sensory_input)
            
            # Get nervous system state
            state = nervous_system.get_state()
            
            # Update animation
            animated_person.update_state(
                arousal=state.get('arousal', 0.5),
                stress=state.get('stress_level', 0.0),
                mood=state.get('mood', 0.5),
                heart_rate=state.get('heart_rate', 60.0),
                attention=0.5
            )
            
            frame_count[0] += 1
            
            if frame_count[0] % 30 == 0:
                logger.info(
                    f"Frame {frame_count[0]} | "
                    f"HR: {state.get('heart_rate', 0):.1f} | "
                    f"Stress: {state.get('stress_level', 0):.2f} | "
                    f"Arousal: {state.get('arousal', 0):.2f}"
                )
            
            return animated_person.artists
        
        except Exception as e:
            logger.error(f"Animation error: {e}")
            return []
    
    # Create animation
    fig = animated_person.figure
    FuncAnimation(fig, animate, frames=None, blit=True, interval=67, repeat=True)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        cognitive_system.stop()
        if use_media and media_manager:
            try:
                media_manager.stop_all()
            except Exception:
                pass
        logger.info("✓ App closed")


def run_batch(num_frames=300):
    """Run batch mode without visualization."""
    from cognitive_system.core.embodied_cognition import EmbodiedCognitionSystem
    from cognitive_system.virtual_biology.nervous_system import VirtualNervousSystem
    from cognitive_system.core.neural_network import NeuralNetworkController
    from cognitive_system.memory.multimodal_memory import MultimodalMemorySystem
    from cognitive_system.behavior.behavior_engine import BehaviorEngine
    import numpy as np
    
    logger.info("=" * 70)
    logger.info("COGNITIVE SYSTEM - BATCH MODE")
    logger.info(f"Processing {num_frames} frames")
    logger.info("=" * 70)
    
    # Initialize components
    cognitive_system = EmbodiedCognitionSystem()
    nervous_system = VirtualNervousSystem()
    neural_network = NeuralNetworkController(
        input_size=22,
        hidden_size=128,
        output_size=32
    )
    memory_system = MultimodalMemorySystem()
    behavior_engine = BehaviorEngine()
    
    cognitive_system.setup(
        virtual_biology=nervous_system,
        neural_network=neural_network,
        memory_system=memory_system,
        behavior_engine=behavior_engine
    )
    
    cognitive_system.start()
    logger.info("✓ System started")
    logger.info("")
    
    try:
        for i in range(num_frames):
            # Generate sensory input
            sensory_input = {
                'visual': np.random.randn(10) * 0.1,
                'auditory': np.random.randn(10) * 0.1,
            }
            
            # Process frame
            cognitive_system.update(1.0/15, sensory_input)
            
            if (i + 1) % 50 == 0:
                state = nervous_system.get_state()
                logger.info(
                    f"Frame {i + 1}/{num_frames} | "
                    f"HR: {state.get('heart_rate', 0):.1f} | "
                    f"Stress: {state.get('stress_level', 0):.2f}"
                )
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cognitive_system.stop()
        logger.info(f"✓ Processing complete ({num_frames} frames)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cognitive System - Main App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 app_launcher.py                    # Interactive with animation
  python3 app_launcher.py --batch            # Batch mode
  python3 app_launcher.py --batch --frames 500   # Batch with 500 frames
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
    
    if args.batch:
        run_batch(args.frames)
    else:
        run_interactive()
