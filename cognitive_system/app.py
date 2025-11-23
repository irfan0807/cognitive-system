"""
Unified Cognitive System App

Single entry point for running the complete cognitive system with:
- Live video and audio feed processing
- Real-time animation of the embodied character
- Neural network decision making
- Memory consolidation
"""

import numpy as np
import sys
import logging
import threading
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CognitiveSystemApp:
    """
    Unified app combining video/audio input with animated character visualization.
    """
    
    def __init__(self, use_camera: bool = True, use_microphone: bool = True, 
                 enable_animation: bool = True, fps: int = 15):
        """
        Initialize the cognitive system app.
        
        Args:
            use_camera: Enable camera input
            use_microphone: Enable microphone input
            enable_animation: Enable character animation
            fps: Frames per second for processing
        """
        logger.info("Initializing Cognitive System App...")
        
        self.use_camera = use_camera
        self.use_microphone = use_microphone
        self.enable_animation = enable_animation
        self.fps = fps
        self.target_frame_time = 1.0 / fps
        
        # Try to import required modules
        try:
            from cognitive_system.core.embodied_cognition import EmbodiedCognitionSystem
            from cognitive_system.virtual_biology.nervous_system import VirtualNervousSystem
            from cognitive_system.core.neural_network import NeuralNetworkController
            from cognitive_system.memory.multimodal_memory import MultimodalMemorySystem
            from cognitive_system.behavior.behavior_engine import BehaviorEngine
            from cognitive_system.utils.media_capture import MediaStreamManager
            from cognitive_system.utils.media_processor import MediaInputBridge
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
        
        # Initialize media capture if needed
        if use_camera or use_microphone:
            logger.info("Initializing media capture...")
            self.media_manager = MediaStreamManager()
            
            if use_camera:
                try:
                    self.media_manager.add_camera_stream(
                        name="camera",
                        camera_id=0,
                        fps=fps
                    )
                    logger.info("✓ Camera stream added")
                except Exception as e:
                    logger.warning(f"Camera unavailable: {e}")
                    self.use_camera = False
            
            if use_microphone:
                try:
                    self.media_manager.add_microphone_stream(
                        name="microphone",
                        sample_rate=16000,
                        chunk_size=1024
                    )
                    logger.info("✓ Microphone stream added")
                except Exception as e:
                    logger.warning(f"Microphone unavailable: {e}")
                    self.use_microphone = False
            
            self.media_bridge = MediaInputBridge()
        else:
            self.media_manager = None
            self.media_bridge = None
        
        # Initialize embodied cognition system
        logger.info("Initializing embodied cognition system...")
        self.cognitive_system = EmbodiedCognitionSystem()
        
        self.nervous_system = VirtualNervousSystem()
        self.neural_network = NeuralNetworkController(
            input_size=22,
            hidden_size=128,
            output_size=32
        )
        self.memory_system = MultimodalMemorySystem()
        self.behavior_engine = BehaviorEngine()
        
        self.cognitive_system.setup(
            virtual_biology=self.nervous_system,
            neural_network=self.neural_network,
            memory_system=self.memory_system,
            behavior_engine=self.behavior_engine
        )
        
        logger.info("✓ Cognitive system initialized")
        
        # Initialize animation if enabled
        if enable_animation:
            try:
                from cognitive_system.visualization import AnimatedPerson
                logger.info("Initializing animation...")
                self.animated_person = AnimatedPerson(figsize=(12, 8))
                logger.info("✓ Animation initialized")
            except Exception as e:
                logger.warning(f"Animation unavailable: {e}")
                self.enable_animation = False
                self.animated_person = None
        else:
            self.animated_person = None
        
        # System state
        self.is_running = False
        self.frame_count = 0
        self.last_sensory_input = None
        self.last_system_state = None
    
    def start_media_streams(self) -> bool:
        """
        Start media capture streams.
        
        Returns:
            True if streams started successfully
        """
        if not self.media_manager:
            logger.info("No media manager configured")
            return False
        
        logger.info("Starting media streams...")
        try:
            success = self.media_manager.start_all()
            if success:
                logger.info("✓ Media streams started")
                return True
            else:
                logger.error("Failed to start media streams")
                return False
        except Exception as e:
            logger.error(f"Error starting streams: {e}")
            return False
    
    def stop_media_streams(self):
        """Stop media capture streams."""
        if self.media_manager:
            logger.info("Stopping media streams...")
            self.media_manager.stop_all()
            logger.info("✓ Media streams stopped")
    
    def get_sensory_input(self) -> Dict[str, Any]:
        """
        Get sensory input from cameras/microphones or generate synthetic input.
        
        Returns:
            Dictionary with sensory inputs
        """
        if self.media_manager and (self.use_camera or self.use_microphone):
            # Try to get real input from media streams
            try:
                sensory_input = {}
                
                if self.use_camera:
                    video_frame = self.media_manager.get_latest_frame("camera")
                    if video_frame is not None:
                        # Extract visual features
                        visual_features = self.media_bridge.process_video(video_frame)
                        sensory_input['visual'] = visual_features[:10]
                    else:
                        sensory_input['visual'] = np.random.randn(10) * 0.1
                else:
                    sensory_input['visual'] = np.random.randn(10) * 0.1
                
                if self.use_microphone:
                    audio_chunk = self.media_manager.get_latest_frame("microphone")
                    if audio_chunk is not None:
                        # Extract audio features
                        audio_features = self.media_bridge.process_audio(audio_chunk)
                        sensory_input['auditory'] = audio_features[:10]
                    else:
                        sensory_input['auditory'] = np.random.randn(10) * 0.1
                else:
                    sensory_input['auditory'] = np.random.randn(10) * 0.1
                
                return sensory_input
            except Exception as e:
                logger.debug(f"Error getting sensory input: {e}")
                return self._generate_synthetic_input()
        else:
            return self._generate_synthetic_input()
    
    def _generate_synthetic_input(self) -> Dict[str, Any]:
        """Generate synthetic sensory input for testing."""
        return {
            'visual': np.random.randn(10) * 0.1,
            'auditory': np.random.randn(10) * 0.1,
        }
    
    def process_frame(self) -> Dict[str, Any]:
        """
        Process one frame of sensory input through the cognitive system.
        
        Returns:
            Dictionary with system outputs
        """
        # Get sensory input
        sensory_input = self.get_sensory_input()
        self.last_sensory_input = sensory_input
        
        # Update cognitive system
        delta_time = self.target_frame_time
        output = self.cognitive_system.update(delta_time, sensory_input)
        
        self.last_system_state = output
        self.frame_count += 1
        
        return output
    
    def run_interactive(self):
        """Run the app with interactive animation."""
        if not self.enable_animation:
            logger.error("Animation not enabled")
            return
        
        logger.info("Starting interactive app with animation...")
        self.is_running = True
        self.cognitive_system.start()
        
        # Start media streams
        if self.media_manager:
            if not self.start_media_streams():
                logger.warning("Could not start media streams, using synthetic input")
        
        # Create figure for animation
        fig = self.animated_person.figure
        
        def animate_frame(frame_num):
            """Animation update function."""
            if not self.is_running:
                return []
            
            try:
                # Process cognitive system
                output = self.process_frame()
                
                # Extract state for visualization
                nervous_state = self.nervous_system.get_state()
                
                # Update animation
                self.animated_person.update_state(
                    arousal=nervous_state.get('arousal', 0.5),
                    stress=nervous_state.get('stress_level', 0.0),
                    mood=nervous_state.get('mood', 0.5),
                    heart_rate=nervous_state.get('heart_rate', 60.0),
                    attention=output.get('attention_level', 0.5) if output else 0.5
                )
                
                # Print status every 30 frames
                if self.frame_count % 30 == 0:
                    logger.info(
                        f"Frame {self.frame_count} | "
                        f"HR: {nervous_state.get('heart_rate', 0):.1f} | "
                        f"Stress: {nervous_state.get('stress_level', 0):.2f} | "
                        f"Arousal: {nervous_state.get('arousal', 0):.2f}"
                    )
                
                return self.animated_person.artists
            
            except Exception as e:
                logger.error(f"Error in animation frame: {e}")
                return []
        
        # Create animation
        anim = FuncAnimation(
            fig,
            animate_frame,
            frames=None,
            blit=True,
            interval=self.target_frame_time * 1000,
            repeat=True
        )
        
        logger.info("Animation started! Close window to exit.")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def run_batch(self, num_frames: int = 300):
        """
        Run the app for a fixed number of frames without visualization.
        
        Args:
            num_frames: Number of frames to process
        """
        logger.info(f"Running batch mode for {num_frames} frames...")
        self.is_running = True
        self.cognitive_system.start()
        
        # Start media streams
        if self.media_manager:
            if not self.start_media_streams():
                logger.warning("Could not start media streams, using synthetic input")
        
        try:
            for i in range(num_frames):
                output = self.process_frame()
                
                if (i + 1) % 50 == 0:
                    nervous_state = self.nervous_system.get_state()
                    logger.info(
                        f"Frame {i + 1}/{num_frames} | "
                        f"HR: {nervous_state.get('heart_rate', 0):.1f} | "
                        f"Stress: {nervous_state.get('stress_level', 0):.2f}"
                    )
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the app."""
        logger.info("Stopping cognitive system app...")
        self.is_running = False
        self.cognitive_system.stop()
        self.stop_media_streams()
        logger.info(f"✓ App stopped (processed {self.frame_count} frames)")


def main():
    """Main entry point for the app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cognitive System - Complete App")
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="Run mode: interactive (with animation) or batch (no visualization)"
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        default=True,
        help="Enable camera input"
    )
    parser.add_argument(
        "--no-camera",
        action="store_false",
        dest="camera",
        help="Disable camera input"
    )
    parser.add_argument(
        "--microphone",
        action="store_true",
        default=True,
        help="Enable microphone input"
    )
    parser.add_argument(
        "--no-microphone",
        action="store_false",
        dest="microphone",
        help="Disable microphone input"
    )
    parser.add_argument(
        "--animation",
        action="store_true",
        default=True,
        help="Enable animation"
    )
    parser.add_argument(
        "--no-animation",
        action="store_false",
        dest="animation",
        help="Disable animation"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of frames (batch mode only)"
    )
    
    args = parser.parse_args()
    
    # Create app
    app = CognitiveSystemApp(
        use_camera=args.camera,
        use_microphone=args.microphone,
        enable_animation=args.animation,
        fps=args.fps
    )
    
    # Run app
    if args.mode == "interactive":
        app.run_interactive()
    else:
        app.run_batch(num_frames=args.frames)


if __name__ == "__main__":
    main()
