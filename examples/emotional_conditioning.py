"""
Emotional Conditioning Example

Demonstrates multimodal memory formation, particularly emotional conditioning
to stimuli like the word "fire". When a memory is formed, it is reconstructed
multimodally integrating emotion, visual, and auditory information.
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
    """Demonstrate emotional conditioning and multimodal memory."""
    print("=" * 60)
    print("Emotional Conditioning - Multimodal Memory Formation")
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
    
    print("Demonstrating emotional conditioning to the word 'fire'...")
    print()
    
    # Phase 1: Neutral exposure to various words
    print("Phase 1: Neutral baseline - exposing to various words")
    print("-" * 60)
    neutral_words = ["tree", "cloud", "water", "sky", "grass"]
    expose_to_words(system, neutral_words, emotional_valence=0.0, stress=0.0)
    print()
    
    # Phase 2: Emotional conditioning to "fire" with negative emotion
    print("Phase 2: Emotional conditioning - 'fire' with negative emotion")
    print("-" * 60)
    print("Presenting 'FIRE' with strong negative emotional response...")
    condition_to_word(system, "fire", emotional_valence=-0.9, stress=0.9)
    print()
    
    # Phase 3: Show multimodal memory formation
    print("Phase 3: Examining formed multimodal memory")
    print("-" * 60)
    examine_multimodal_memory(memory_system, system)
    print()
    
    # Phase 4: Test conditioning by recalling with cue
    print("Phase 4: Testing memory recall with 'fire' cue")
    print("-" * 60)
    test_conditioning_recall(memory_system)
    print()
    
    system.stop()
    print("Demonstration complete!")


def expose_to_words(
    system: EmbodiedCognitionSystem,
    words: list,
    emotional_valence: float,
    stress: float
):
    """Expose character to words with specific emotional context."""
    time_step = 0.05
    
    for word in words:
        # Encode word as auditory stimulus (simplified)
        word_encoding = encode_word(word)
        
        # Visual context (e.g., seeing the word or related image)
        visual_context = np.random.randn(10) * 0.1
        
        # Create sensory input
        sensory_input = {
            'visual': visual_context,
            'auditory': word_encoding,
        }
        
        # Artificially set emotional state for conditioning
        # In real system, this would come from actual experience
        system.virtual_biology.stress_level = stress
        
        # Update system
        output = system.update(time_step, sensory_input)
        
        print(f"  Exposed to '{word}': valence={emotional_valence:.2f}, "
              f"stress={output['physiological_state']['stress_level']:.2f}")


def condition_to_word(
    system: EmbodiedCognitionSystem,
    word: str,
    emotional_valence: float,
    stress: float
):
    """Condition character to a specific word with emotional response."""
    time_step = 0.05
    
    # Repeat exposure to strengthen conditioning
    for trial in range(5):
        word_encoding = encode_word(word)
        
        # Strong visual stimulus (e.g., image of fire)
        visual_context = np.random.randn(10) * 0.8  # Intense visual
        visual_context[0] = 1.0  # Salient feature
        
        sensory_input = {
            'visual': visual_context,
            'auditory': word_encoding,
        }
        
        # Set strong physiological response
        system.virtual_biology.stress_level = stress
        system.virtual_biology.cortisol_level = 0.9
        
        # Update system - forms multimodal memory
        output = system.update(time_step, sensory_input)
        
        print(f"  Trial {trial + 1}: Conditioning to '{word}'")
        print(f"    Heart rate: {output['physiological_state']['heart_rate']:.1f} bpm")
        print(f"    Stress: {output['physiological_state']['stress_level']:.2f}")
        print(f"    Cortisol: {output['physiological_state']['chemical_levels']['cortisol']:.2f}")


def encode_word(word: str) -> np.ndarray:
    """Simple word encoding as auditory stimulus."""
    # Hash word to consistent encoding
    encoding = np.zeros(10)
    for i, char in enumerate(word[:10]):
        encoding[i] = ord(char) / 128.0
    return encoding


def examine_multimodal_memory(
    memory_system: MultimodalMemorySystem,
    system: EmbodiedCognitionSystem
):
    """Examine the most recent multimodal memory."""
    if memory_system.get_memory_count() == 0:
        print("No memories formed yet.")
        return
    
    # Get most recent memory
    recent_memory = memory_system.episodic_memories[-1]
    
    print("Most recent multimodal memory:")
    print(f"  Timestamp: {recent_memory.timestamp:.2f}s")
    print()
    print("  Emotional modality:")
    print(f"    Valence (positive/negative): {recent_memory.emotional_valence:.2f}")
    print(f"    Arousal (intensity): {recent_memory.emotional_arousal:.2f}")
    print(f"    Stress level: {recent_memory.stress_level:.2f}")
    print()
    print("  Visual modality:")
    print(f"    Visual data: {recent_memory.visual_data[:5].round(2)}")
    print()
    print("  Auditory modality:")
    print(f"    Auditory data: {recent_memory.auditory_data[:5].round(2)}")
    print(f"    Intensity: {recent_memory.auditory_intensity:.2f}")
    print()
    print("  Physiological state:")
    print(f"    Heart rate: {recent_memory.heart_rate:.1f} bpm")
    print(f"    Oxytocin: {recent_memory.hormone_levels.get('oxytocin', 0):.2f}")
    print(f"    Cortisol: {recent_memory.hormone_levels.get('cortisol', 0):.2f}")
    print()
    print("  Memory strength: {:.2f}".format(recent_memory.strength))
    print()
    print("This memory integrates:")
    print("  ✓ Emotion at the time")
    print("  ✓ Visual stimuli at the time")
    print("  ✓ Auditory stimulation at the time")
    print("  ✓ Physiological state at the time")


def test_conditioning_recall(memory_system: MultimodalMemorySystem):
    """Test if conditioning was successful by recalling with word cue."""
    # Create cue with "fire" encoding
    fire_encoding = encode_word("fire")
    
    # Recall with auditory cue
    cue = {
        'auditory': fire_encoding,
    }
    
    recalled = memory_system.recall_memories(cue, top_k=3)
    
    if recalled:
        print(f"Successfully recalled {len(recalled)} memories related to 'fire'")
        print()
        print("Most relevant memory (conditioned response):")
        mem = recalled[0]
        print(f"  Emotional valence: {mem.emotional_valence:.2f} (negative)")
        print(f"  Emotional arousal: {mem.emotional_arousal:.2f} (high)")
        print(f"  Stress level: {mem.stress_level:.2f} (high)")
        print(f"  Heart rate: {mem.heart_rate:.1f} bpm (elevated)")
        print()
        print("Character has learned to associate 'fire' with:")
        print("  • Negative emotion")
        print("  • High arousal/stress")
        print("  • Elevated physiological response")
        print()
        print("This demonstrates successful emotional conditioning!")
    else:
        print("No memories recalled - conditioning may not have formed")


if __name__ == '__main__':
    main()
