#!/usr/bin/env python3
"""
Cognitive System - Animation with Video Feed (Original Style)

This uses the original animated person and adds live video feed on the side.

Layout:
- LEFT: Live Video Feed (640x480)
- MIDDLE: Animated Character (original animation)  
- RIGHT: Physiological Metrics

Usage:
    python3 app_with_video.py          # With camera
    python3 app_with_video.py --test   # Test mode (no camera)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import logging
import matplotlib
# Use MacOSX backend for better macOS compatibility
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cognitive System with Animation + Video Feed")
    parser.add_argument("--test", action="store_true", help="Test mode (no camera)")
    args = parser.parse_args()
    
    # Import components
    from cognitive_system.core.embodied_cognition import EmbodiedCognitionSystem
    from cognitive_system.virtual_biology.nervous_system import VirtualNervousSystem
    from cognitive_system.core.neural_network import NeuralNetworkController
    from cognitive_system.memory.multimodal_memory import MultimodalMemorySystem
    from cognitive_system.behavior.behavior_engine import BehaviorEngine
    from cognitive_system.visualization import AnimatedPerson
    from cognitive_system.rag.multimodal_rag import MultimodalRAGSystem
    from cognitive_system.utils.system_monitor import SystemMonitor
    
    logger.info("=" * 70)
    logger.info("COGNITIVE SYSTEM - ANIMATION WITH LIVE VIDEO FEED")
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
    
    # Initialize RAG system for retrieval-based responses
    logger.info("Initializing RAG system...")
    rag_system = MultimodalRAGSystem(
        embedding_dim=128,
        visualizer_update_interval=0.5
    )
    rag_system.attach_cognitive_system(cognitive_system)
    rag_system.start()
    logger.info("✓ RAG system initialized and started")
    
    # Initialize system monitor for debugging
    logger.info("Initializing system monitor...")
    monitor = SystemMonitor(history_size=100, verbose=True)
    logger.info("✓ System monitor ready")
    
    # Initialize animation with video support
    logger.info("Initializing animation with video feed...")
    animated_person = AnimatedPerson(figsize=(15, 8), with_video=(not args.test))
    logger.info("✓ Animation ready")
    
    # Initialize media capture
    media_manager = None
    if not args.test:
        try:
            from cognitive_system.utils.media_capture import MediaStreamManager
            
            logger.info("Initializing camera...")
            media_manager = MediaStreamManager()
            
            try:
                media_manager.add_camera_stream(name="camera", camera_id=0, fps=15)
                media_manager.start_all()
                logger.info("✓ Camera started")
            except Exception as e:
                logger.warning(f"Camera not available: {e}")
                media_manager = None
        
        except ImportError:
            logger.warning("Media capture not available")
            media_manager = None
    
    # Start cognitive system
    cognitive_system.start()
    logger.info("✓ Cognitive system started")
    logger.info("")
    logger.info("Animation running... Close window to exit")
    logger.info("")
    
    # State
    frame_count = [0]
    
    def animate(frame_num):
        """Update animation frame."""
        try:
            # Start monitoring this frame
            monitor.start_frame(frame_count[0])
            
            # Get video frame
            frame = None
            if media_manager and not args.test:
                try:
                    frame = media_manager.get_frame("camera")
                    if frame is not None:
                        animated_person.update_video_frame(frame)
                        monitor.record_video_frame(frame)
                        
                        # Process frame through RAG system
                        visual_features = rag_system.process_video_frame(frame, store_embedding=True)
                        monitor.record_visual_features(visual_features)
                except Exception as e:
                    logger.debug(f"Video frame error: {e}")
                    visual_features = np.random.randn(128) * 0.1
            else:
                # Test mode: synthetic visual features
                visual_features = np.random.randn(128) * 0.1
            
            # Generate audio features (synthetic in test mode)
            audio_features = np.random.randn(128) * 0.1
            
            # Retrieve context from RAG based on current features
            rag_context = None
            if not args.test:
                try:
                    rag_context = rag_system.retrieve_context(
                        query_embedding=visual_features,
                        top_k=5
                    )
                    if rag_context:
                        monitor.record_rag_retrieval(rag_context)
                except Exception as e:
                    logger.debug(f"RAG retrieval error: {e}")
            
            # Update RAG cognitive state
            rag_system.update_cognitive_state(
                visual_features=visual_features,
                audio_features=audio_features,
                rag_context=rag_context
            )
            
            # Generate sensory input
            sensory_input = {
                'visual': visual_features[:10],
                'auditory': audio_features[:10],
            }
            
            # Update system
            cognitive_system.update(1.0/15, sensory_input)
            monitor.record_neural_network(sensory_input['visual'])
            
            # Get state
            state = nervous_system.get_state()
            
            # Blend with RAG state for enhanced cognitive response
            if rag_context:
                context_weight = rag_context.get('context_relevance', 0.0)
                state['attention'] = 0.5 + 0.5 * context_weight
            
            # Update animation
            animation_state = {
                'arousal': state.get('arousal', 0.5),
                'stress': state.get('stress_level', 0.0),
                'mood': state.get('mood', 0.5),
                'heart_rate': state.get('heart_rate', 60.0),
                'attention': state.get('attention', 0.5),
                'emotional_valence': 0.0,
                'emotional_arousal': 0.5,
                'oxytocin': 0.5
            }
            
            animated_person.update_state(animation_state)
            monitor.record_animation_update(animation_state)
            
            # Get animation elements to display
            artists = animated_person.update()
            
            frame_count[0] += 1
            
            # End frame monitoring
            monitor.end_frame()
            
            if frame_count[0] % 30 == 0:
                logger.info("")
                logger.info(monitor.get_status_string())
            
            return artists
        
        except Exception as e:
            logger.error(f"Animation error: {e}", exc_info=True)
            monitor.end_frame()
            return []
    
    # Create animation
    fig = animated_person.fig
    logger.info("Creating FuncAnimation...")
    anim = FuncAnimation(fig, animate, frames=None, blit=False, interval=67, repeat=True, cache_frame_data=False)
    logger.info("✓ Animation created")
    
    try:
        plt.tight_layout()
        logger.info("Displaying window...")
        plt.show()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        logger.info("Shutting down...")
        cognitive_system.stop()
        rag_system.stop()
        
        if media_manager:
            try:
                media_manager.stop_all()
                logger.info("✓ Camera stopped")
            except Exception:
                pass
        
        # Print monitoring report
        monitor.print_report()
        
        logger.info("✓ App closed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
