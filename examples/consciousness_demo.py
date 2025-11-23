"""
Interactive Demo: Deep Neural Network Consciousness System

Demonstrates the consciousness simulation system with live video and audio feeds.
Shows how deep neural networks integrate with embodied cognition for
consciousness-like behavior.
"""

import numpy as np
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def run_simulated_demo():
    """Run demo with simulated video and audio (no camera/microphone needed)."""
    from cognitive_system.core.conscious_cognition import ConsciousCognitionSystem
    
    print("\n" + "=" * 70)
    print("CONSCIOUSNESS SIMULATION - Simulated Feeds Demo")
    print("=" * 70)
    print("\nThis demo uses simulated video and audio feeds.")
    print("No camera or microphone required.\n")
    
    # Initialize system without live feeds
    system = ConsciousCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01,
        use_live_feeds=False  # Use simulated feeds
    )
    
    print("System initialized successfully!")
    print("\nRunning consciousness simulation for 5 seconds...")
    print("Processing simulated video and audio through deep neural networks...\n")
    
    # Run demonstration
    system.demonstrate_consciousness(duration=5.0)
    
    print("\n✓ Demo completed successfully!")


def run_live_demo():
    """Run demo with live camera and microphone feeds."""
    from cognitive_system.core.conscious_cognition import ConsciousCognitionSystem
    
    print("\n" + "=" * 70)
    print("CONSCIOUSNESS SIMULATION - Live Feeds Demo")
    print("=" * 70)
    print("\nThis demo uses live camera and microphone feeds.")
    print("Make sure you have a camera and microphone connected.\n")
    
    # Initialize system with live feeds
    system = ConsciousCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01,
        use_live_feeds=True,
        use_camera=True,
        use_microphone=True
    )
    
    print("System initialized successfully!")
    
    # Start live feeds
    print("\nStarting live feeds...")
    if system.start_live_feeds():
        print("✓ Live feeds started!")
    else:
        print("⚠ Could not start live feeds, falling back to simulated feeds")
    
    print("\nRunning consciousness simulation for 10 seconds...")
    print("Processing live video and audio through deep neural networks...\n")
    
    # Run demonstration
    try:
        system.demonstrate_consciousness(duration=10.0)
    finally:
        system.stop_live_feeds()
    
    print("\n✓ Demo completed successfully!")


def run_interactive_demo():
    """Run interactive demo with detailed consciousness monitoring."""
    from cognitive_system.core.conscious_cognition import ConsciousCognitionSystem
    import time
    
    print("\n" + "=" * 70)
    print("INTERACTIVE CONSCIOUSNESS SIMULATION")
    print("=" * 70)
    print("\nThis demo processes sensory input and shows how the")
    print("consciousness network integrates with virtual biology.\n")
    
    # Initialize system
    system = ConsciousCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01,
        use_live_feeds=False
    )
    
    print("System initialized!\n")
    
    # Scenario 1: Calm observation
    print("=" * 70)
    print("Scenario 1: Calm Environmental Observation")
    print("-" * 70)
    print("Processing calm visual and auditory environment...\n")
    
    for i in range(3):
        output = system.process_live_consciousness()
        
        print(f"Step {i + 1}:")
        print(f"  Consciousness state dimension: {output['consciousness_features']['consciousness_state'].shape}")
        print(f"  Visual features: {output['consciousness_features']['visual_features'].shape}")
        print(f"  Auditory features: {output['consciousness_features']['auditory_features'].shape}")
        print(f"  Arousal modulation: {output['integrated_state']['arousal_modulation']:.3f}")
        print(f"  Attention focus: {output['integrated_state']['attention_focus']:.3f}")
        print(f"  Physiological arousal: {output['physiological_state']['arousal']:.3f}")
        print(f"  Heart rate: {output['physiological_state']['heart_rate']:.1f} bpm")
        print()
        time.sleep(0.5)
    
    # Scenario 2: Processing with social interaction
    print("=" * 70)
    print("Scenario 2: Social Interaction with High Engagement")
    print("-" * 70)
    print("Processing environment with social stimuli...\n")
    
    for i in range(3):
        output = system.process_live_consciousness()
        
        # Simulate social interaction
        state = system.process_sensory_input(
            visual_input=output['consciousness_features']['visual_features'].flatten()[:64],
            auditory_input=output['consciousness_features']['auditory_features'].flatten()[:32],
            social_context=0.9,  # High social engagement
            threat_level=0.0,
            reward_signal=0.7
        )
        
        print(f"Step {i + 1}:")
        print(f"  Emotional valence: {state.emotional_state[0]:.3f} (positive)")
        print(f"  Emotional arousal: {state.emotional_state[1]:.3f}")
        print(f"  Oxytocin level: {output['physiological_state']['oxytocin']:.3f}")
        print(f"  Stress level: {output['physiological_state']['stress']:.3f}")
        print(f"  Mood: {system.get_state_summary()['mood']:.3f}")
        print()
        time.sleep(0.5)
    
    # Scenario 3: Memory formation
    print("=" * 70)
    print("Scenario 3: Memory Formation from Conscious Experience")
    print("-" * 70)
    print("Processing emotionally significant stimuli...\n")
    
    initial_memories = system.get_state_summary()['total_memories']
    
    for i in range(5):
        output = system.process_live_consciousness()
        
        # Simulate emotionally significant event
        state = system.process_sensory_input(
            visual_input=output['consciousness_features']['visual_features'].flatten()[:64],
            auditory_input=output['consciousness_features']['auditory_features'].flatten()[:32],
            social_context=0.5,
            threat_level=0.2,
            reward_signal=0.6
        )
        
        current_memories = system.get_state_summary()['total_memories']
        
        if current_memories > initial_memories:
            print(f"✓ New memory formed! (Total: {current_memories})")
            initial_memories = current_memories
        
        time.sleep(0.3)
    
    print()
    
    # Show final summary
    print("=" * 70)
    print("Final System State")
    print("-" * 70)
    summary = system.get_state_summary()
    print(f"Total memories formed: {summary['total_memories']}")
    print(f"Current arousal: {summary['arousal']:.3f}")
    print(f"Current mood: {summary['mood']:.3f}")
    print(f"Current stress: {summary['stress']:.3f}")
    print(f"Heart rate: {summary['heart_rate']:.1f} bpm")
    print()
    
    # Test decision making influenced by consciousness
    print("=" * 70)
    print("Consciousness-Driven Decision Making")
    print("-" * 70)
    
    actions = ["explore environment", "rest", "social approach", "vigilant monitoring"]
    action, confidence = system.make_decision(actions)
    
    print(f"Available actions: {actions}")
    print(f"Selected action: '{action}'")
    print(f"Decision confidence: {confidence:.3f}")
    print(f"\n(Decision influenced by deep neural consciousness state,")
    print(f" integrated with virtual biology and emotional state)")
    print()
    
    print("=" * 70)
    print("✓ Interactive demo completed!")
    print("=" * 70)


def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description='Deep Neural Network Consciousness System Demo'
    )
    parser.add_argument(
        '--mode',
        choices=['simulated', 'live', 'interactive'],
        default='simulated',
        help='Demo mode: simulated (no camera/mic), live (with camera/mic), or interactive'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("COGNITIVE SYSTEM - DEEP NEURAL NETWORK CONSCIOUSNESS")
    print("=" * 70)
    print("\nThis system demonstrates consciousness simulation by integrating:")
    print("  • Deep CNN for visual cortex (hierarchical feature extraction)")
    print("  • Deep RNN for auditory cortex (temporal audio processing)")
    print("  • Multimodal integration network (consciousness workspace)")
    print("  • Virtual biology (brain structures, physiology, neurotransmitters)")
    print("  • Embodied cognition (biology drives cognition)")
    print("  • Real-time learning and memory formation")
    print()
    
    try:
        if args.mode == 'simulated':
            run_simulated_demo()
        elif args.mode == 'live':
            run_live_demo()
        elif args.mode == 'interactive':
            run_interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
