"""
Simple Animated Person Demo

This is a simplified version that generates static images showing the animated
person in different states. Run this to see how the character responds to the
neural network without needing an interactive display.
"""

from cognitive_system.embodied_cognition import EmbodiedCognitionSystem
from cognitive_system.visualization import AnimatedPerson
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import os


def main():
    """Generate visualizations of animated person in different states."""
    print("=" * 70)
    print("Animated Person - Simple Demo")
    print("=" * 70)
    print()
    print("Generating visualizations of the animated person connected to the")
    print("neural network in different cognitive states...")
    print()
    
    # Create output directory
    output_dir = "animated_person_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize cognitive system
    system = EmbodiedCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01
    )
    
    # Define scenarios to visualize
    scenarios = [
        {
            'name': 'Calm & Happy',
            'description': 'Low stress, positive social interaction',
            'visual_intensity': 0.3,
            'auditory_intensity': 0.3,
            'social_context': 0.8,
            'threat_level': 0.0,
            'reward_signal': 0.7,
            'filename': 'calm_happy.png'
        },
        {
            'name': 'Stressed & Anxious',
            'description': 'High threat, no social support',
            'visual_intensity': 0.8,
            'auditory_intensity': 0.8,
            'social_context': 0.0,
            'threat_level': 0.9,
            'reward_signal': 0.0,
            'filename': 'stressed_anxious.png'
        },
        {
            'name': 'Social Support',
            'description': 'High threat buffered by social bonding',
            'visual_intensity': 0.7,
            'auditory_intensity': 0.7,
            'social_context': 0.9,
            'threat_level': 0.8,
            'reward_signal': 0.3,
            'filename': 'social_support.png'
        },
        {
            'name': 'Moderate Arousal',
            'description': 'Engaged and attentive state',
            'visual_intensity': 0.5,
            'auditory_intensity': 0.5,
            'social_context': 0.5,
            'threat_level': 0.3,
            'reward_signal': 0.6,
            'filename': 'moderate_arousal.png'
        }
    ]
    
    # Generate visualization for each scenario
    for i, scenario in enumerate(scenarios):
        print(f"{i+1}. {scenario['name']}")
        print(f"   {scenario['description']}")
        
        # Create new person for each scenario
        person = AnimatedPerson(figsize=(14, 8))
        
        # Generate sensory input
        visual_input = np.random.randn(64) * scenario['visual_intensity']
        auditory_input = np.random.randn(32) * scenario['auditory_intensity']
        
        # Process through cognitive system
        state = system.process_sensory_input(
            visual_input=visual_input,
            auditory_input=auditory_input,
            social_context=scenario['social_context'],
            threat_level=scenario['threat_level'],
            reward_signal=scenario['reward_signal']
        )
        
        # Get system state
        summary = system.get_state_summary()
        
        # Update animated person with neural network state
        person.update_state(summary)
        person.update(frame=i * 5)  # Different animation frame for each
        
        # Save visualization
        filepath = os.path.join(output_dir, scenario['filename'])
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Print state info
        print(f"   Heart Rate: {summary['heart_rate']:.0f} bpm")
        print(f"   Arousal: {summary['arousal']:.2f}")
        print(f"   Stress: {summary['stress']:.2f}")
        print(f"   Mood: {summary['mood']:.2f}")
        print(f"   Saved to: {filepath}")
        print()
    
    print("=" * 70)
    print(f"✓ Generated {len(scenarios)} visualizations")
    print(f"✓ Output directory: {output_dir}/")
    print("✓ The animated person is connected to the neural network!")
    print("  - Character appearance changes based on emotional state")
    print("  - Body movements reflect arousal and stress levels")
    print("  - Heart rate shown through pulsing visualization")
    print("  - Real-time metrics displayed on the right panel")
    print("=" * 70)


if __name__ == '__main__':
    main()
