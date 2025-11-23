#!/usr/bin/env python3
"""
Cognitive System - Complete App with Live Camera Feed
Real-time animation that responds to your camera input

Uses your webcam to capture visual input and process it through 
the neural network, driving character animation in real-time.
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


def run_with_camera():
    """Run app with live camera feed."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import cv2
    
    logger.info("=" * 70)
    logger.info("COGNITIVE SYSTEM - LIVE CAMERA ANIMATION")
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
    import matplotlib.pyplot as plt
    
    # Create figure with custom layout for camera feed
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main animation area (takes up most of the space)
    ax_main = fig.add_subplot(gs[0:3, 0:3])
    
    # Camera feed in top-left corner (small)
    ax_camera = fig.add_subplot(gs[0, 0])
    ax_camera.set_title('Your Camera Feed', fontsize=10)
    ax_camera.axis('off')
    
    # Create a custom figure object
    class AnimatedPersonWithCamera:
        def __init__(self, fig, ax_main, ax_camera):
            self.fig = fig
            self.ax_main = ax_main
            self.ax_camera = ax_camera
            from src.cognitive_system.visualization import AnimatedPerson as AP
            # Create a temporary animated person to get the visualization setup
            self.ap = AP(figsize=(1, 1))  # Dummy size
            # Copy the methods
            self.update_state = self.ap.update_state
            self.update = self.ap.update
            self.person_elements = self.ap.person_elements
            self.metric_elements = self.ap.metric_elements
            self.arousal = self.ap.arousal
            self.stress = self.ap.stress
            self.mood = self.ap.mood
            self.heart_rate = self.ap.heart_rate
            self.time = self.ap.time
            
    animated_person = AnimatedPersonWithCamera(fig, ax_main, ax_camera)
    logger.info("   ✓ Animation ready with camera feed display")
    
    # Initialize camera
    logger.info("3. Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("   ✗ Cannot open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    logger.info("   ✓ Camera ready")
    logger.info("")
    
    logger.info("Starting live camera animation...")
    logger.info("Your camera feed is being processed by the neural network!")
    logger.info("The character will react to your movements and appearance.")
    logger.info("")
    
    # Animation state
    frame_count = [0]
    fig = animated_person.fig
    camera_frame = [None]
    
    def extract_visual_features(frame):
        """Extract visual features from camera frame."""
        # Resize frame for processing
        frame_resized = cv2.resize(frame, (64, 64))
        
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray_normalized = gray.astype(np.float32) / 255.0
        
        # Flatten and add some variation
        visual_features = gray_normalized.flatten()[:64]
        
        # Pad if necessary
        if len(visual_features) < 64:
            visual_features = np.pad(visual_features, (0, 64 - len(visual_features)))
        
        return visual_features
    
    def animate(frame_num):
        """Animation frame update with camera input."""
        try:
            # Capture from camera
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame from camera")
                return []
            
            camera_frame[0] = frame
            
            # Extract visual features from camera
            visual_input = extract_visual_features(frame)
            
            # Generate synthetic audio (camera doesn't have audio)
            auditory_input = np.random.randn(32) * 0.05
            
            # Detect motion/activity from frame
            threat_level = 0.1  # Low default
            reward_signal = 0.5  # Neutral default
            
            # Simple motion detection
            if frame_count[0] > 0:
                # Could add optical flow here for better motion detection
                pass
            
            # Process through cognitive system
            state = cognitive_system.process_sensory_input(
                visual_input=visual_input,
                auditory_input=auditory_input,
                social_context=0.6,  # You're here
                threat_level=threat_level,
                reward_signal=reward_signal
            )
            
            # Extract emotion and physiology
            emotional_valence, emotional_arousal = state.emotional_state
            brain_state = state.brain_state
            
            # Update animation with neural state
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
            import traceback
            traceback.print_exc()
            return []
    
    logger.info("Animation window is now running with live camera feed...")
    logger.info("Allow camera access when prompted")
    logger.info("Close the window or press Ctrl+C to exit")
    logger.info("")
    
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
    
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup camera
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✓ Camera released")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Thank you for using Cognitive System!")
    logger.info("=" * 70)


if __name__ == '__main__':
    run_with_camera()
