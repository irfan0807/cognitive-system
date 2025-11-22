"""
Example usage of the Embodied Cognition System

This script demonstrates the autonomous AI with embodied cognition,
showing how virtual biology drives virtual cognition.
"""

import numpy as np
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem


def run_example():
    """Run a demonstration of the embodied cognition system."""
    print("=" * 70)
    print("Embodied Cognition System - Demonstration")
    print("=" * 70)
    print()
    
    # Initialize the system
    print("Initializing embodied cognition system...")
    system = EmbodiedCognitionSystem(
        visual_dim=64,
        auditory_dim=32,
        attention_dim=128,
        learning_rate=0.01
    )
    print("✓ System initialized")
    print()
    
    # Scenario 1: Calm, positive social interaction
    print("Scenario 1: Calm, positive social interaction")
    print("-" * 70)
    
    visual_input = np.random.randn(64) * 0.3  # Mild visual stimulus
    auditory_input = np.random.randn(32) * 0.3  # Mild auditory stimulus (e.g., friendly voice)
    
    state = system.process_sensory_input(
        visual_input=visual_input,
        auditory_input=auditory_input,
        social_context=0.8,  # High social interaction
        threat_level=0.0,    # No threat
        reward_signal=0.7    # Positive experience
    )
    
    summary = system.get_state_summary()
    print(f"  Arousal Level: {summary['arousal']:.3f}")
    print(f"  Heart Rate: {summary['heart_rate']:.1f} bpm")
    print(f"  Oxytocin (social bonding): {summary['oxytocin']:.3f}")
    print(f"  Stress Response: {summary['stress']:.3f}")
    print(f"  Mood (serotonin): {summary['mood']:.3f}")
    print(f"  Emotional Valence: {summary['emotional_valence']:.3f} (positive)")
    print(f"  Emotional Arousal: {summary['emotional_arousal']:.3f}")
    print()
    
    # Scenario 2: Stressful, threatening situation
    print("Scenario 2: Stressful, threatening situation")
    print("-" * 70)
    
    visual_input = np.random.randn(64) * 0.8  # Intense visual stimulus
    auditory_input = np.random.randn(32) * 0.8  # Loud auditory stimulus
    
    state = system.process_sensory_input(
        visual_input=visual_input,
        auditory_input=auditory_input,
        social_context=0.0,  # No social support
        threat_level=0.9,    # High threat
        reward_signal=0.0    # No positive reinforcement
    )
    
    summary = system.get_state_summary()
    print(f"  Arousal Level: {summary['arousal']:.3f}")
    print(f"  Heart Rate: {summary['heart_rate']:.1f} bpm")
    print(f"  Oxytocin (social bonding): {summary['oxytocin']:.3f}")
    print(f"  Stress Response: {summary['stress']:.3f}")
    print(f"  Attention Level: {summary['attention_level']:.3f}")
    print(f"  Emotional Valence: {summary['emotional_valence']:.3f} (negative)")
    print(f"  Emotional Arousal: {summary['emotional_arousal']:.3f}")
    print()
    
    # Scenario 3: Social buffering - threat with social support
    print("Scenario 3: Social buffering - threat with social support")
    print("-" * 70)
    print("(Demonstrates how oxytocin from social bonding reduces stress)")
    
    visual_input = np.random.randn(64) * 0.7
    auditory_input = np.random.randn(32) * 0.7
    
    state = system.process_sensory_input(
        visual_input=visual_input,
        auditory_input=auditory_input,
        social_context=0.9,  # High social support
        threat_level=0.8,    # High threat (but with support)
        reward_signal=0.3    # Some positive reinforcement from support
    )
    
    summary = system.get_state_summary()
    print(f"  Arousal Level: {summary['arousal']:.3f}")
    print(f"  Heart Rate: {summary['heart_rate']:.1f} bpm")
    print(f"  Oxytocin (social bonding): {summary['oxytocin']:.3f} (elevated)")
    print(f"  Stress Response: {summary['stress']:.3f} (buffered by oxytocin)")
    print(f"  Homeostatic Balance: {summary['homeostasis']:.3f}")
    print()
    
    # Decision making
    print("Decision Making - Biology Drives Cognition")
    print("-" * 70)
    
    possible_actions = ["approach", "avoid", "explore", "rest"]
    action, confidence = system.make_decision(possible_actions)
    
    print(f"  Available actions: {possible_actions}")
    print(f"  Selected action: '{action}' (confidence: {confidence:.3f})")
    print(f"  (Decision influenced by current biological and emotional state)")
    print()
    
    # Memory formation and retrieval
    print("Memory System")
    print("-" * 70)
    
    summary = system.get_state_summary()
    print(f"  Total memories formed: {summary['total_memories']}")
    
    if summary['total_memories'] > 0:
        # Try to recall similar experiences
        similar = system.recall_similar_experiences(
            emotional_query=(summary['emotional_valence'], summary['emotional_arousal']),
            top_k=2
        )
        print(f"  Recalled {len(similar)} similar experiences")
        for i, (memory, similarity) in enumerate(similar):
            print(f"    Memory {i+1}: Valence={memory.emotional_valence:.2f}, "
                  f"Arousal={memory.emotional_arousal:.2f}, "
                  f"Similarity={similarity:.3f}")
    print()
    
    # Show how biology drives cognition
    print("Virtual Biology → Virtual Cognition Integration")
    print("-" * 70)
    print("Brain Structures:")
    print(f"  • Brain Stem → Arousal: {summary['arousal']:.3f}")
    print(f"  • Pituitary Gland → Oxytocin: {summary['oxytocin']:.3f}")
    print(f"  • Paraventricular Nucleus → Stress: {summary['stress']:.3f}")
    print()
    print("Physiology → Cognitive Modulation:")
    print(f"  • Norepinephrine → Attention: {summary['attention_level']:.3f}")
    print(f"  • Serotonin → Mood: {summary['mood']:.3f}")
    print(f"  • Dopamine → Motivation: {summary['motivation']:.3f}")
    print()
    print("Multimodal Memory:")
    print(f"  • Emotion + Visual + Auditory → Integrated Memories")
    print(f"  • Total: {summary['total_memories']} memories")
    print()
    
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_example()
