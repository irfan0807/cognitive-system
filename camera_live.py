#!/usr/bin/env python3
"""
Cognitive System - Live Camera + Animation
Camera feed displayed at top-left of animation window
Character responds to your real-time movements
"""

import numpy as np
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_with_camera_overlay():
    """Run app with camera feed overlaid on animation."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import cv2
    
    logger.info("=" * 70)
    logger.info("COGNITIVE SYSTEM - LIVE CAMERA + ANIMATION")
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
    animated_person = AnimatedPerson(figsize=(16, 9))
    logger.info("   ✓ Animation ready")
    
    # Initialize camera
    logger.info("3. Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("   ✗ Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    logger.info("   ✓ Camera ready")
    logger.info("")
    
    logger.info("Starting live camera animation...")
    logger.info("Camera feed displayed at TOP-LEFT corner")
    logger.info("Character reacts to your movements!")
    logger.info("")
    
    # Animation state
    frame_count = [0]
    fig = animated_person.fig
    camera_image = [None]
    camera_axes = [None]
    
    # Add camera feed display area to the figure
    ax_camera = fig.add_axes([0.02, 0.73, 0.15, 0.25])  # [left, bottom, width, height]
    ax_camera.set_title('📷 Camera Feed', fontsize=9, fontweight='bold')
    ax_camera.axis('off')
    im = ax_camera.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
    camera_axes[0] = (ax_camera, im)
    
    def extract_visual_features(frame):
        """Extract visual features from camera frame."""
        frame_resized = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray_normalized = gray.astype(np.float32) / 255.0
        visual_features = gray_normalized.flatten()[:64]
        
        if len(visual_features) < 64:
            visual_features = np.pad(visual_features, (0, 64 - len(visual_features)))
        
        return visual_features
    
    def animate(frame_num):
        """Animation frame update with camera input."""
        try:
            # Capture from camera
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                return []
            
            # Flip for mirror effect (more natural)
            frame = cv2.flip(frame, 1)
            
            # Update camera display
            if camera_axes[0]:
                ax_cam, im_cam = camera_axes[0]
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_cam.set_data(frame_rgb)
            
            # Extract visual features
            visual_input = extract_visual_features(frame)
            
            # Synthetic audio
            auditory_input = np.random.randn(32) * 0.05
            
            # Process through cognitive system
            state = cognitive_system.process_sensory_input(
                visual_input=visual_input,
                auditory_input=auditory_input,
                social_context=0.6,
                threat_level=0.1,
                reward_signal=0.5
            )
            
            # Extract state
            emotional_valence, emotional_arousal = state.emotional_state
            brain_state = state.brain_state
            
            # Update animation
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
            
            # Status updates
            if frame_count[0] % 30 == 0:
                logger.info(
                    f"Frame {frame_count[0]:4d} | "
                    f"HR: {brain_state.get('heart_rate', 0):5.1f} | "
                    f"Arousal: {emotional_arousal:.2f} | "
                    f"Mood: {emotional_valence:.2f}"
                )
            
            # Return animated artists
            artists = animated_person.update(frame_count[0])
            artists.append(camera_axes[0][1])  # Add camera image to update list
            return artists
        
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    logger.info("Animation window is running...")
    logger.info("Close window or press Ctrl+C to exit")
    logger.info("")
    
    # Create animation
    anim = FuncAnimation(
        fig,
        animate,
        frames=None,
        blit=True,
        interval=67,
        repeat=True,
        cache_frame_data=False
    )
    
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✓ Camera released")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Thank you!")
    logger.info("=" * 70)


if __name__ == '__main__':
    run_with_camera_overlay()
