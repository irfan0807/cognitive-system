"""
Environmental Interaction Example

This example demonstrates the character's ability to interact with its
environment, including:
- Watching videos
- Playing games
- Responding to music (automatically responding to the beat)
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

logging.basicConfig(level=logging.INFO)


def main():
    """Demonstrate environmental interaction capabilities."""
    print("=" * 60)
    print("Environmental Interaction Demonstration")
    print("=" * 60)
    print()
    
    # Initialize system
    nervous_system = VirtualNervousSystem()
    brain = VirtualBrain()
    memory_system = MultimodalMemorySystem()
    behavior_engine = BehaviorEngine()
    neural_network = NeuralNetworkController(22, 50, 20)
    
    system = EmbodiedCognitionSystem()
    system.setup(nervous_system, neural_network, memory_system, behavior_engine)
    system.start()
    
    print("System initialized. Starting demonstrations...")
    print()
    
    # 1. Music Response Demonstration
    print("-" * 60)
    print("1. Music Response - Character responds to the beat")
    print("-" * 60)
    demonstrate_music_response(system, nervous_system, behavior_engine)
    
    # 2. Video Watching Demonstration
    print("\n" + "-" * 60)
    print("2. Video Watching - Character attends to visual content")
    print("-" * 60)
    demonstrate_video_watching(system, behavior_engine)
    
    # 3. Game Playing Demonstration
    print("\n" + "-" * 60)
    print("3. Game Playing - Character interacts with game")
    print("-" * 60)
    demonstrate_game_playing(system, behavior_engine)
    
    system.stop()
    print("\nDemonstration complete!")


def demonstrate_music_response(
    system: EmbodiedCognitionSystem,
    nervous_system: VirtualNervousSystem,
    behavior_engine: BehaviorEngine
):
    """
    Demonstrate the character automatically responding to music beat.
    
    The character's stimulus system is affected by music, generating
    corresponding movement patterns in the body.
    """
    print("Playing music with rhythmic beat...")
    
    time_step = 0.016
    beat_frequency = 2.0  # 2 Hz beat (120 BPM)
    
    for step in range(100):
        # Generate musical beat
        time = step * time_step
        beat_phase = (time * beat_frequency) % 1.0
        beat_strength = 1.0 if beat_phase < 0.1 else 0.0
        
        # Music affects the nervous system
        nervous_system.respond_to_music(beat_strength, time_step)
        
        # Behavior engine responds to music
        music_data = {
            'beat_strength': beat_strength,
            'beat_frequency': beat_frequency,
        }
        interaction = behavior_engine.interact_with_environment('music', music_data)
        
        # Show response on strong beats
        if beat_strength > 0.5 and step % 31 == 0:  # ~Every half second
            print(f"  Beat detected! Movement synchronization: {interaction['beat_synchronization']:.2f}")
            rhythmic = interaction['rhythmic_movement']
            print(f"    Rhythmic movement intensity: {np.mean(np.abs(rhythmic)):.3f}")
    
    print(f"Total beats: {int(100 * time_step * beat_frequency)}")
    print("Character exhibited rhythmic movement patterns synchronized to beat")


def demonstrate_video_watching(
    system: EmbodiedCognitionSystem,
    behavior_engine: BehaviorEngine
):
    """Demonstrate character watching video and tracking visual content."""
    print("Showing video with varying visual saliency...")
    
    time_step = 0.016
    
    for step in range(100):
        # Simulate video with regions of varying saliency
        # (e.g., moving object, face, etc.)
        saliency_map = np.random.rand(10)
        saliency_map[step % 10] = 1.0  # One highly salient region
        
        video_data = {
            'saliency': saliency_map,
            'frame': step,
        }
        
        interaction = behavior_engine.interact_with_environment('video', video_data)
        
        # Update system with visual input
        sensory_input = {
            'visual': saliency_map,
            'auditory': np.zeros(10),
        }
        system.update(time_step, sensory_input)
        
        # Report attention periodically
        if step % 25 == 0:
            engagement = interaction['engagement']
            print(f"  Frame {step}: Engagement level: {engagement:.2f}")
            print(f"    Attention focused on region: {np.argmax(saliency_map)}")
    
    print("Character tracked salient features throughout video")


def demonstrate_game_playing(
    system: EmbodiedCognitionSystem,
    behavior_engine: BehaviorEngine
):
    """Demonstrate character playing a simple game."""
    print("Character playing simple action game...")
    
    time_step = 0.016
    action_space = ['move_left', 'move_right', 'jump', 'duck', 'attack']
    score = 0
    
    for step in range(100):
        # Simulate game state
        game_state = {
            'player_position': step % 10,
            'enemy_position': (step + 5) % 10,
            'score': score,
        }
        
        game_data = {
            'state': game_state,
            'action_space': action_space,
        }
        
        # Character interacts with game
        interaction = behavior_engine.interact_with_environment('game', game_data)
        
        # Update system
        sensory_input = {
            'visual': np.array([game_state['player_position'], 
                              game_state['enemy_position']] + [0] * 8),
            'auditory': np.zeros(10),
        }
        system.update(time_step, sensory_input)
        
        # Process action
        action = interaction.get('action')
        if action:
            # Simple scoring: reward any action
            score += 1
            
            if step % 25 == 0:
                print(f"  Step {step}: Action: {action}, Score: {score}")
    
    print(f"Game complete! Final score: {score}")
    print("Character autonomously selected actions based on game state")


if __name__ == '__main__':
    main()
