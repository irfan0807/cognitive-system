"""
Basic Example - Creating an Autonomous Character

This example demonstrates how to create a basic autonomous character using
the Cognitive System framework, with all components integrated.
"""

import numpy as np
import logging

from cognitive_system import (
    EmbodiedCognitionSystem,
    VirtualNervousSystem,
    VirtualBrain,
    MultimodalMemorySystem,
    BehaviorEngine
)
from cognitive_system.core.neural_network import NeuralNetworkController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Main simulation loop."""
    print("=" * 60)
    print("Cognitive System - Autonomous Character Simulation")
    print("=" * 60)
    print()
    
    # Create system components
    print("Initializing system components...")
    
    nervous_system = VirtualNervousSystem()
    brain = VirtualBrain()
    memory_system = MultimodalMemorySystem()
    behavior_engine = BehaviorEngine()
    
    # Neural network: 22 inputs (10 visual + 10 auditory + 2 physiological)
    #                50 hidden units
    #                20 outputs (motor commands + attention)
    neural_network = NeuralNetworkController(
        input_size=22,
        hidden_size=50,
        output_size=20
    )
    
    # Create main embodied cognition system
    system = EmbodiedCognitionSystem()
    system.setup(nervous_system, neural_network, memory_system, behavior_engine)
    
    print("System initialized successfully!")
    print()
    
    # Start simulation
    system.start()
    print("Starting simulation...")
    print()
    
    # Simulation parameters
    time_step = 0.016  # ~60 FPS
    total_steps = 500
    
    # Run simulation
    for step in range(total_steps):
        # Generate sensory input (simulated environment)
        sensory_input = generate_sensory_input(step, total_steps)
        
        # Update the system
        output = system.update(delta_time=time_step, sensory_input=sensory_input)
        
        # Learn from experiences periodically
        if step % 50 == 0 and step > 0:
            system.learn_from_experience(learning_rate=0.01)
            
            # Print status
            state = system.get_state()
            print(f"Step {step}/{total_steps}:")
            print(f"  Simulation time: {state['simulation_time']:.2f}s")
            print(f"  Experiences: {state['experience_count']}")
            print(f"  Memories: {state['memory_count']}")
            print(f"  Heart rate: {output['physiological_state']['heart_rate']:.1f} bpm")
            print(f"  Stress: {output['physiological_state']['stress_level']:.2f}")
            print(f"  Arousal: {output['physiological_state']['arousal']:.2f}")
            print()
    
    # Final statistics
    print("=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    final_state = system.get_state()
    print(f"Total simulation time: {final_state['simulation_time']:.2f}s")
    print(f"Total experiences: {final_state['experience_count']}")
    print(f"Total memories: {final_state['memory_count']}")
    print()
    
    # Test memory recall
    print("Testing memory recall...")
    test_memory_recall(memory_system)
    
    # Stop system
    system.stop()
    print("System stopped.")


def generate_sensory_input(step: int, total_steps: int) -> dict:
    """
    Generate simulated sensory input for the character.
    
    This simulates varying environmental conditions to demonstrate
    the character's responses.
    """
    # Time-varying stimuli
    t = step / total_steps
    
    # Visual input (varies sinusoidally)
    visual = np.sin(t * 2 * np.pi * 3) * 0.3 + 0.5
    visual_features = np.random.randn(10) * 0.1 + visual
    
    # Auditory input (periodic beats for music response)
    beat_frequency = 2.0  # Hz
    beat_phase = (step * 0.016 * beat_frequency) % 1.0
    auditory_beat = 1.0 if beat_phase < 0.1 else 0.0
    auditory_features = np.random.randn(10) * 0.1 + auditory_beat * 0.5
    
    # Simulate stressful event at certain points
    stress_event = 1.0 if 100 < step < 120 or 300 < step < 320 else 0.0
    
    return {
        'visual': visual_features,
        'auditory': auditory_features,
        'stress_trigger': stress_event,
    }


def test_memory_recall(memory_system: MultimodalMemorySystem):
    """Test the multimodal memory recall system."""
    if memory_system.get_memory_count() == 0:
        print("No memories to recall.")
        return
    
    # Try to recall memories with different cues
    print("\n1. Recalling memories with visual cue...")
    visual_cue = {'visual': np.random.randn(10) * 0.1}
    visual_memories = memory_system.recall_memories(visual_cue, top_k=3)
    print(f"   Recalled {len(visual_memories)} memories")
    
    print("\n2. Recalling memories with emotional cue...")
    emotional_cue = {
        'emotional_state': {'valence': -0.5, 'arousal': 0.8}  # High arousal negative
    }
    emotional_memories = memory_system.recall_memories(emotional_cue, top_k=3)
    print(f"   Recalled {len(emotional_memories)} memories")
    
    if emotional_memories:
        print(f"\n   Most relevant memory:")
        mem = emotional_memories[0]
        print(f"     Timestamp: {mem.timestamp:.2f}s")
        print(f"     Emotional valence: {mem.emotional_valence:.2f}")
        print(f"     Emotional arousal: {mem.emotional_arousal:.2f}")
        print(f"     Memory strength: {mem.strength:.2f}")
    
    print("\n3. Recalling memories with multimodal cue...")
    multimodal_cue = {
        'visual': np.random.randn(10) * 0.1,
        'auditory': np.random.randn(10) * 0.1,
        'emotional_state': {'valence': 0.3, 'arousal': 0.6}
    }
    multimodal_memories = memory_system.recall_memories(multimodal_cue, top_k=3)
    print(f"   Recalled {len(multimodal_memories)} memories")


if __name__ == '__main__':
    main()
