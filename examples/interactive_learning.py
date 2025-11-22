"""
Interactive Learning Example

Demonstrates how the embodied cognition system learns from experiences
and how biological state influences learning and decision-making.
"""

import numpy as np
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem


def run_learning_example():
    """Demonstrate learning through repeated experiences."""
    print("=" * 70)
    print("Interactive Learning Example")
    print("Virtual Biology Drives Learning and Adaptation")
    print("=" * 70)
    print()
    
    system = EmbodiedCognitionSystem(learning_rate=0.05)  # Higher learning rate for demo
    
    print("Scenario: Agent learns to respond to different situations")
    print()
    
    # Define some consistent situations
    situations = {
        'safe_social': {
            'visual': np.random.randn(64) * 0.3,
            'auditory': np.random.randn(32) * 0.3,
            'social': 0.9,
            'threat': 0.0,
            'reward': 0.8,
            'description': 'Safe social environment'
        },
        'threatening': {
            'visual': np.random.randn(64) * 0.8,
            'auditory': np.random.randn(32) * 0.8,
            'social': 0.0,
            'threat': 0.9,
            'reward': 0.0,
            'description': 'Threatening environment'
        },
        'uncertain': {
            'visual': np.random.randn(64) * 0.5,
            'auditory': np.random.randn(32) * 0.5,
            'social': 0.3,
            'threat': 0.5,
            'reward': 0.2,
            'description': 'Uncertain environment'
        }
    }
    
    actions = ['approach', 'avoid', 'explore', 'rest']
    
    print("Phase 1: Initial Responses (before learning)")
    print("-" * 70)
    
    initial_responses = {}
    for sit_name, situation in situations.items():
        state = system.process_sensory_input(
            visual_input=situation['visual'],
            auditory_input=situation['auditory'],
            social_context=situation['social'],
            threat_level=situation['threat'],
            reward_signal=situation['reward']
        )
        
        action, confidence = system.make_decision(actions)
        initial_responses[sit_name] = action
        
        summary = system.get_state_summary()
        print(f"\n{situation['description']}:")
        print(f"  Biological State:")
        print(f"    - Arousal: {summary['arousal']:.3f}")
        print(f"    - Oxytocin: {summary['oxytocin']:.3f}")
        print(f"    - Stress: {summary['stress']:.3f}")
        print(f"    - Mood: {summary['mood']:.3f}")
        print(f"  Cognitive State:")
        print(f"    - Attention: {summary['attention_level']:.3f}")
        print(f"    - Motivation: {summary['motivation']:.3f}")
        print(f"  Decision: {action} (confidence: {confidence:.3f})")
    
    print()
    print("\nPhase 2: Learning Through Experience")
    print("-" * 70)
    print("Exposing the system to repeated experiences...")
    print()
    
    # Simulate multiple exposures with feedback
    for episode in range(5):
        print(f"Episode {episode + 1}:")
        
        for sit_name, situation in situations.items():
            # Process the situation
            state = system.process_sensory_input(
                visual_input=situation['visual'],
                auditory_input=situation['auditory'],
                social_context=situation['social'],
                threat_level=situation['threat'],
                reward_signal=situation['reward']
            )
            
            action, confidence = system.make_decision(actions)
            
            # Provide feedback based on the action appropriateness
            # In a safe social environment, 'approach' is good
            # In threatening environment, 'avoid' is good
            if sit_name == 'safe_social' and action == 'approach':
                outcome_reward = 1.0
            elif sit_name == 'threatening' and action == 'avoid':
                outcome_reward = 1.0
            elif sit_name == 'uncertain' and action in ['explore', 'rest']:
                outcome_reward = 0.5
            else:
                outcome_reward = -0.5
            
            # Learn from the outcome
            system.learn_from_feedback(
                previous_input={'visual': situation['visual'], 'auditory': situation['auditory']},
                action_taken=action,
                outcome_reward=outcome_reward
            )
            
            print(f"  {sit_name}: {action} → reward: {outcome_reward:+.1f}")
        
        print()
    
    print("Phase 3: Post-Learning Responses")
    print("-" * 70)
    
    # Test again after learning
    for sit_name, situation in situations.items():
        state = system.process_sensory_input(
            visual_input=situation['visual'],
            auditory_input=situation['auditory'],
            social_context=situation['social'],
            threat_level=situation['threat'],
            reward_signal=situation['reward']
        )
        
        action, confidence = system.make_decision(actions)
        
        summary = system.get_state_summary()
        print(f"\n{situation['description']}:")
        print(f"  Initial Action: {initial_responses[sit_name]}")
        print(f"  Learned Action: {action} (confidence: {confidence:.3f})")
        print(f"  Biological Influence:")
        print(f"    - Stress level affecting decision: {summary['stress']:.3f}")
        print(f"    - Oxytocin (social bonding): {summary['oxytocin']:.3f}")
        print(f"    - Motivation: {summary['motivation']:.3f}")
    
    print()
    print("\nPhase 4: Memory Analysis")
    print("-" * 70)
    
    summary = system.get_state_summary()
    print(f"Total memories formed: {summary['total_memories']}")
    
    # Show how different emotional states are remembered
    print("\nMemories of positive experiences:")
    positive_memories = system.recall_similar_experiences(
        emotional_query=(0.8, 0.5),  # Positive valence, moderate arousal
        top_k=3
    )
    
    for i, (memory, similarity) in enumerate(positive_memories):
        print(f"  Memory {i+1}:")
        print(f"    - Emotional Valence: {memory.emotional_valence:.2f}")
        print(f"    - Emotional Arousal: {memory.emotional_arousal:.2f}")
        print(f"    - Similarity to query: {similarity:.3f}")
        print(f"    - Memory strength: {memory.strength:.3f}")
    
    print("\nMemories of negative/threatening experiences:")
    negative_memories = system.recall_similar_experiences(
        emotional_query=(-0.8, 0.8),  # Negative valence, high arousal
        top_k=3
    )
    
    for i, (memory, similarity) in enumerate(negative_memories):
        print(f"  Memory {i+1}:")
        print(f"    - Emotional Valence: {memory.emotional_valence:.2f}")
        print(f"    - Emotional Arousal: {memory.emotional_arousal:.2f}")
        print(f"    - Similarity to query: {similarity:.3f}")
        print(f"    - Memory strength: {memory.strength:.3f}")
    
    print()
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Biology drives cognition: Stress, oxytocin, and neurotransmitters")
    print("   directly influence attention and decision-making")
    print()
    print("2. Real-time learning: The system adapts through experience,")
    print("   learning which actions work in which biological states")
    print()
    print("3. Multimodal memory: Experiences are stored with emotional,")
    print("   visual, auditory, and physiological context")
    print()
    print("4. Emotional consolidation: Emotionally significant events")
    print("   (both positive and negative) are remembered more strongly")
    print()
    print("5. Social buffering: Oxytocin from social bonding reduces")
    print("   stress responses, enabling better decision-making")
    print("=" * 70)


if __name__ == "__main__":
    run_learning_example()
