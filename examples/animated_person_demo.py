"""
Animated Person Example

This example demonstrates the animated person visualization connected to the
neural network. The character's appearance and movements respond in real-time
to the cognitive system's state.
"""

import numpy as np
import matplotlib.pyplot as plt
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem
from cognitive_system.visualization import AnimatedPerson
import logging

logging.basicConfig(level=logging.INFO)


def main():
    """Run the animated person demonstration."""
    print("=" * 70)
    print("Animated Person - Neural Network Connected Visualization")
    print("=" * 70)
    print()
    
    # Initialize the cognitive system
    print("Initializing embodied cognition system...")
    system = EmbodiedCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01
    )
    print("✓ Cognitive system initialized")
    
    # Initialize the animated person
    print("Creating animated person visualization...")
    person = AnimatedPerson(figsize=(14, 8))
    print("✓ Animated person created")
    print()
    
    print("Running animation scenarios...")
    print("The character will respond to different neural network states:")
    print("  1. Calm, positive social interaction")
    print("  2. Stressful, threatening situation")
    print("  3. Social buffering with support")
    print()
    
    # Define scenarios
    scenarios = [
        {
            'name': 'Calm Social Interaction',
            'visual_intensity': 0.3,
            'auditory_intensity': 0.3,
            'social_context': 0.8,
            'threat_level': 0.0,
            'reward_signal': 0.7,
            'duration': 100
        },
        {
            'name': 'Stressful Threat',
            'visual_intensity': 0.8,
            'auditory_intensity': 0.8,
            'social_context': 0.0,
            'threat_level': 0.9,
            'reward_signal': 0.0,
            'duration': 100
        },
        {
            'name': 'Social Buffering',
            'visual_intensity': 0.7,
            'auditory_intensity': 0.7,
            'social_context': 0.9,
            'threat_level': 0.8,
            'reward_signal': 0.3,
            'duration': 100
        }
    ]
    
    # Create animation callback
    scenario_idx = [0]  # Use list to allow modification in nested function
    frame_count = [0]
    current_scenario = scenarios[0]
    
    def update_animation(frame):
        """Update function for animation that cycles through scenarios."""
        nonlocal current_scenario
        
        # Check if we need to switch scenarios
        if frame_count[0] >= current_scenario['duration']:
            scenario_idx[0] = (scenario_idx[0] + 1) % len(scenarios)
            current_scenario = scenarios[scenario_idx[0]]
            frame_count[0] = 0
            print(f"\n→ Scenario: {current_scenario['name']}")
        
        frame_count[0] += 1
        
        # Generate sensory input for current scenario
        visual_input = np.random.randn(64) * current_scenario['visual_intensity']
        auditory_input = np.random.randn(32) * current_scenario['auditory_intensity']
        
        # Process through cognitive system
        state = system.process_sensory_input(
            visual_input=visual_input,
            auditory_input=auditory_input,
            social_context=current_scenario['social_context'],
            threat_level=current_scenario['threat_level'],
            reward_signal=current_scenario['reward_signal']
        )
        
        # Get system state summary
        summary = system.get_state_summary()
        
        # Update animated person with neural network state
        person.update_state(summary)
        
        # Update animation
        return person.update(frame)
    
    print("Starting animation (close window to exit)...")
    print("Watch how the character responds to different scenarios!")
    print()
    
    # Create and display animation
    from matplotlib.animation import FuncAnimation
    
    anim = FuncAnimation(
        person.fig,
        update_animation,
        frames=300,  # Total frames (will cycle through scenarios)
        interval=50,  # 50ms between frames = 20 FPS
        blit=False
    )
    
    plt.tight_layout()
    plt.show()
    
    print()
    print("=" * 70)
    print("Animation complete!")
    print("=" * 70)


def demonstrate_static_states():
    """Demonstrate static visualization of different states."""
    print("=" * 70)
    print("Static State Demonstrations")
    print("=" * 70)
    print()
    
    # Initialize system
    system = EmbodiedCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01
    )
    
    states_to_show = [
        {
            'name': 'Calm & Happy',
            'arousal': 0.3,
            'stress': 0.2,
            'mood': 0.9,
            'heart_rate': 65.0,
            'emotional_valence': 0.8,
            'emotional_arousal': 0.3
        },
        {
            'name': 'Stressed & Anxious',
            'arousal': 0.9,
            'stress': 0.9,
            'mood': 0.3,
            'heart_rate': 100.0,
            'emotional_valence': -0.7,
            'emotional_arousal': 0.9
        }
    ]
    
    for state_config in states_to_show:
        print(f"\nShowing state: {state_config['name']}")
        
        person = AnimatedPerson(figsize=(14, 8))
        person.update_state(state_config)
        person.show()


if __name__ == '__main__':
    # Run the main animated demonstration
    main()
    
    # Optionally, uncomment to see static demonstrations
    # demonstrate_static_states()
