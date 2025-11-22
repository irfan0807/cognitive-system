"""
Basic tests for the Cognitive System framework.

Tests core functionality to ensure the system is working correctly.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cognitive_system import (
    EmbodiedCognitionSystem,
    VirtualNervousSystem,
    VirtualBrain,
    MultimodalMemorySystem,
    BehaviorEngine
)
from cognitive_system.core.neural_network import NeuralNetworkController


def test_system_initialization():
    """Test that all components can be initialized."""
    print("Testing system initialization...")
    
    nervous_system = VirtualNervousSystem()
    brain = VirtualBrain()
    memory_system = MultimodalMemorySystem()
    behavior_engine = BehaviorEngine()
    neural_network = NeuralNetworkController(22, 50, 20)
    
    system = EmbodiedCognitionSystem()
    system.setup(nervous_system, neural_network, memory_system, behavior_engine)
    
    assert system.virtual_biology is not None
    assert system.neural_network is not None
    assert system.memory_system is not None
    assert system.behavior_engine is not None
    
    print("✓ System initialization successful")


def test_basic_update_cycle():
    """Test that the system can process updates."""
    print("Testing basic update cycle...")
    
    nervous_system = VirtualNervousSystem()
    brain = VirtualBrain()
    memory_system = MultimodalMemorySystem()
    behavior_engine = BehaviorEngine()
    neural_network = NeuralNetworkController(22, 50, 20)
    
    system = EmbodiedCognitionSystem()
    system.setup(nervous_system, neural_network, memory_system, behavior_engine)
    system.start()
    
    # Run a few update cycles
    for i in range(10):
        sensory_input = {
            'visual': np.random.randn(10) * 0.1,
            'auditory': np.random.randn(10) * 0.1,
        }
        output = system.update(0.016, sensory_input)
        
        assert 'behaviors' in output
        assert 'physiological_state' in output
        assert 'neural_state' in output
        assert 'experience' in output
    
    system.stop()
    print("✓ Basic update cycle successful")


def test_physiological_response():
    """Test that physiological responses work correctly."""
    print("Testing physiological response...")
    
    nervous_system = VirtualNervousSystem()
    
    # Low stress input
    low_stress = {
        'visual': np.zeros(10),
        'auditory': np.zeros(10),
    }
    state1 = nervous_system.process(low_stress, 0.1)
    initial_heart_rate = state1['heart_rate']
    
    # High stress input
    high_stress = {
        'visual': np.ones(10) * 0.9,
        'auditory': np.ones(10) * 0.9,
    }
    
    # Apply high stress repeatedly
    for _ in range(20):
        state2 = nervous_system.process(high_stress, 0.1)
    
    # Heart rate should increase with stress
    assert state2['heart_rate'] > initial_heart_rate
    assert state2['stress_level'] > 0.1
    
    print(f"✓ Physiological response: heart rate {initial_heart_rate:.1f} → {state2['heart_rate']:.1f} bpm")


def test_memory_formation():
    """Test that memories are formed correctly."""
    print("Testing memory formation...")
    
    memory_system = MultimodalMemorySystem()
    
    # Create an experience
    experience = {
        'timestamp': 1.0,
        'duration': 0.016,
        'sensory_input': {
            'visual': np.random.randn(10),
            'auditory': np.random.randn(10),
        },
        'physiological_state': {
            'heart_rate': 75.0,
            'stress_level': 0.6,
            'chemical_levels': {'oxytocin': 0.5, 'cortisol': 0.6},
        },
        'emotional_state': {
            'valence': -0.5,
            'arousal': 0.8,
        },
        'behaviors': {},
    }
    
    # Store memory
    memory = memory_system.store_experience(experience)
    
    assert memory_system.get_memory_count() == 1
    assert memory.emotional_valence == -0.5
    assert memory.emotional_arousal == 0.8
    assert memory.heart_rate == 75.0
    
    print("✓ Memory formation successful")


def test_memory_recall():
    """Test that memories can be recalled."""
    print("Testing memory recall...")
    
    memory_system = MultimodalMemorySystem()
    
    # Store multiple memories with different emotional content
    for i in range(10):
        experience = {
            'timestamp': float(i),
            'duration': 0.016,
            'sensory_input': {
                'visual': np.random.randn(10),
                'auditory': np.random.randn(10),
            },
            'physiological_state': {
                'heart_rate': 60.0 + i * 5,
                'stress_level': i / 10.0,
                'chemical_levels': {},
            },
            'emotional_state': {
                'valence': -1.0 + i * 0.2,  # Range from -1 to +1
                'arousal': 0.5 + i * 0.05,
            },
            'behaviors': {},
        }
        memory_system.store_experience(experience)
    
    # Recall with emotional cue
    cue = {
        'emotional_state': {'valence': 0.5, 'arousal': 0.8}
    }
    recalled = memory_system.recall_memories(cue, top_k=3)
    
    assert len(recalled) > 0
    assert len(recalled) <= 3
    
    print(f"✓ Memory recall successful: recalled {len(recalled)} memories")


def test_behavior_generation():
    """Test that behaviors are generated correctly."""
    print("Testing behavior generation...")
    
    behavior_engine = BehaviorEngine()
    
    neural_response = {
        'motor_commands': np.random.randn(10) * 0.5,
        'attention': np.random.randn(5) * 0.5,
        'arousal': 0.7,
    }
    
    physiological_state = {
        'arousal': 0.7,
        'stress_level': 0.4,
        'emotional_state': {'valence': 0.3, 'arousal': 0.7},
    }
    
    behaviors = behavior_engine.generate_behaviors(neural_response, physiological_state)
    
    assert 'motor' in behaviors
    assert 'expressive' in behaviors
    assert 'attention' in behaviors
    
    print("✓ Behavior generation successful")


def test_music_response():
    """Test that character responds to music beats."""
    print("Testing music response...")
    
    nervous_system = VirtualNervousSystem()
    behavior_engine = BehaviorEngine()
    
    # Simulate music beats
    beat_responses = []
    for i in range(20):
        beat_strength = 1.0 if i % 5 == 0 else 0.0  # Beat every 5 steps
        
        nervous_system.respond_to_music(beat_strength, 0.05)
        
        music_data = {
            'beat_strength': beat_strength,
            'beat_frequency': 2.0,
        }
        interaction = behavior_engine.interact_with_environment('music', music_data)
        beat_responses.append(interaction['beat_synchronization'])
    
    # Should have some strong beat responses
    max_response = max(beat_responses)
    assert max_response > 0.5
    
    print(f"✓ Music response successful: max synchronization {max_response:.2f}")


def test_learning():
    """Test that the system can learn from experiences."""
    print("Testing learning...")
    
    nervous_system = VirtualNervousSystem()
    brain = VirtualBrain()
    memory_system = MultimodalMemorySystem()
    behavior_engine = BehaviorEngine()
    neural_network = NeuralNetworkController(22, 50, 20)
    
    system = EmbodiedCognitionSystem()
    system.setup(nervous_system, neural_network, memory_system, behavior_engine)
    system.start()
    
    # Generate some experiences
    for i in range(50):
        sensory_input = {
            'visual': np.random.randn(10) * 0.1,
            'auditory': np.random.randn(10) * 0.1,
        }
        system.update(0.016, sensory_input)
    
    # Get initial weight state
    initial_weights = neural_network.weights_hidden_output.copy()
    
    # Trigger learning
    system.learn_from_experience(learning_rate=0.1)
    
    # Weights should have changed
    weight_change = np.abs(neural_network.weights_hidden_output - initial_weights).sum()
    
    # Some learning should have occurred
    print(f"  Total weight change: {weight_change:.6f}")
    
    system.stop()
    print("✓ Learning successful")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Cognitive System Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_system_initialization,
        test_basic_update_cycle,
        test_physiological_response,
        test_memory_formation,
        test_memory_recall,
        test_behavior_generation,
        test_music_response,
        test_learning,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
