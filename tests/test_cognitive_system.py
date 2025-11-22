"""
Tests for the Embodied Cognition System
"""

import numpy as np
import unittest
from cognitive_system.brain import VirtualBrain, BrainStem, PituitaryGland, ParaventricularNucleus
from cognitive_system.neural import RealtimeNeuralNetwork, NeuralLayer
from cognitive_system.memory import MultimodalMemorySystem
from cognitive_system.physiology import VirtualPhysiology, VirtualNervousSystem
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem


class TestBrainStructures(unittest.TestCase):
    """Test virtual brain structures."""
    
    def test_brain_stem_arousal(self):
        """Test brain stem arousal modulation."""
        brainstem = BrainStem()
        
        # High sensory input should increase arousal
        state = brainstem.update(sensory_input=0.9, stress_level=0.1)
        self.assertGreater(state['arousal'], 0.5)
        self.assertGreater(state['heart_rate'], 70.0)
    
    def test_pituitary_oxytocin(self):
        """Test oxytocin release."""
        pituitary = PituitaryGland()
        
        # Social stimulus should increase oxytocin
        initial = pituitary.oxytocin_level
        oxytocin = pituitary.release_oxytocin(social_stimulus=0.9, positive_emotion=0.8)
        self.assertGreater(oxytocin, initial)
    
    def test_pvn_stress_response(self):
        """Test paraventricular nucleus stress response."""
        pvn = ParaventricularNucleus()
        
        # High threat should increase stress
        state = pvn.process_stress(threat_level=0.9, arousal=0.8)
        self.assertGreater(state['stress_response'], 0.3)
        
        # Oxytocin should buffer stress
        modulated = pvn.modulate_oxytocin(oxytocin_level=0.9)
        self.assertLess(modulated, state['stress_response'])
    
    def test_virtual_brain_integration(self):
        """Test integrated virtual brain."""
        brain = VirtualBrain()
        
        state = brain.process(
            sensory_input=0.5,
            social_stimulus=0.8,
            threat_level=0.3,
            positive_emotion=0.7
        )
        
        # Verify all components are present
        self.assertIn('arousal', state)
        self.assertIn('heart_rate', state)
        self.assertIn('oxytocin', state)
        self.assertIn('stress_response', state)


class TestNeuralNetworks(unittest.TestCase):
    """Test neural network components."""
    
    def test_neural_layer_forward(self):
        """Test forward pass through neural layer."""
        layer = NeuralLayer(10, 5)
        x = np.random.randn(2, 10)
        output = layer.forward(x)
        
        self.assertEqual(output.shape, (2, 5))
    
    def test_neural_layer_backward(self):
        """Test backward pass and learning."""
        layer = NeuralLayer(10, 5, learning_rate=0.1)
        x = np.random.randn(2, 10)
        
        # Forward pass
        output = layer.forward(x)
        
        # Backward pass
        gradient = np.random.randn(2, 5)
        initial_weights = layer.weights.copy()
        input_grad = layer.backward(gradient)
        
        # Weights should have changed
        self.assertFalse(np.allclose(layer.weights, initial_weights))
        self.assertEqual(input_grad.shape, (2, 10))
    
    def test_realtime_network_learning(self):
        """Test real-time learning capability."""
        network = RealtimeNeuralNetwork([10, 20, 10], learning_rate=0.01)
        
        x = np.random.randn(1, 10)
        target = np.random.randn(1, 10)
        
        # Train should return a loss value
        loss = network.train_online(x, target)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)


class TestMemorySystem(unittest.TestCase):
    """Test multimodal memory system."""
    
    def test_memory_formation(self):
        """Test memory formation."""
        memory_system = MultimodalMemorySystem(visual_dim=32, auditory_dim=16)
        
        visual = np.random.randn(32)
        auditory = np.random.randn(16)
        
        memory = memory_system.form_memory(
            visual_input=visual,
            auditory_input=auditory,
            emotional_valence=0.8,
            emotional_arousal=0.6
        )
        
        self.assertEqual(memory_system.get_memory_count(), 1)
        self.assertGreater(memory.strength, 0.0)
    
    def test_memory_retrieval(self):
        """Test memory retrieval."""
        memory_system = MultimodalMemorySystem(visual_dim=32, auditory_dim=16)
        
        # Form some memories
        for i in range(3):
            visual = np.random.randn(32)
            auditory = np.random.randn(16)
            memory_system.form_memory(
                visual, auditory,
                emotional_valence=0.5 + i * 0.1,
                emotional_arousal=0.5
            )
        
        # Retrieve similar
        similar = memory_system.retrieve_similar(
            query_emotion=(0.6, 0.5),
            top_k=2
        )
        
        self.assertEqual(len(similar), 2)
        self.assertIsInstance(similar[0][1], float)  # similarity score
    
    def test_memory_decay(self):
        """Test memory decay over time."""
        memory_system = MultimodalMemorySystem(visual_dim=32, auditory_dim=16, decay_rate=0.5)
        
        visual = np.random.randn(32)
        auditory = np.random.randn(16)
        
        memory = memory_system.form_memory(visual, auditory, 0.5, 0.5)
        initial_strength = memory.strength
        
        memory_system.decay_memories()
        
        self.assertLess(memory.strength, initial_strength)


class TestPhysiology(unittest.TestCase):
    """Test virtual physiology and nervous system."""
    
    def test_autonomic_nervous_system(self):
        """Test autonomic nervous system balance."""
        from cognitive_system.physiology import AutonomicNervousSystem
        
        ans = AutonomicNervousSystem()
        
        # High stress should increase sympathetic
        state = ans.process(stress=0.9, relaxation=0.1)
        self.assertGreater(state['sympathetic'], state['parasympathetic'])
        
        # High relaxation should increase parasympathetic
        state = ans.process(stress=0.1, relaxation=0.9)
        self.assertGreater(state['parasympathetic'], state['sympathetic'])
    
    def test_neurotransmitter_modulation(self):
        """Test neurotransmitter modulation."""
        ns = VirtualNervousSystem()
        
        # Reward should increase dopamine
        initial_da = ns.neurotransmitters['dopamine'].level
        ns.modulate_neurotransmitters(reward=0.8)
        self.assertGreater(ns.neurotransmitters['dopamine'].level, initial_da)
    
    def test_virtual_physiology_update(self):
        """Test complete physiology update."""
        physiology = VirtualPhysiology()
        
        brain_state = {
            'arousal': 0.7,
            'stress_response': 0.3,
            'oxytocin': 0.6
        }
        
        sensory_inputs = {
            'visual': np.random.randn(32),
            'auditory': np.random.randn(32)
        }
        
        state = physiology.update(brain_state, sensory_inputs, reward=0.5)
        
        self.assertIn('nervous_system_state', state)
        self.assertIn('energy_level', state)
        self.assertIn('homeostatic_balance', state)
        self.assertIn('cognitive_modulation', state)


class TestEmbodiedCognition(unittest.TestCase):
    """Test the complete embodied cognition system."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        system = EmbodiedCognitionSystem()
        
        self.assertIsNotNone(system.brain)
        self.assertIsNotNone(system.physiology)
        self.assertIsNotNone(system.memory)
        self.assertIsNotNone(system.attention_network)
        self.assertIsNotNone(system.decision_network)
    
    def test_sensory_processing(self):
        """Test sensory input processing."""
        system = EmbodiedCognitionSystem()
        
        visual = np.random.randn(64)
        auditory = np.random.randn(32)
        
        state = system.process_sensory_input(
            visual_input=visual,
            auditory_input=auditory,
            social_context=0.5,
            threat_level=0.2,
            reward_signal=0.7
        )
        
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.brain_state)
        self.assertIsNotNone(state.physiological_state)
        self.assertIsNotNone(state.cognitive_modulation)
    
    def test_decision_making(self):
        """Test decision making."""
        system = EmbodiedCognitionSystem()
        
        # Process some input first
        visual = np.random.randn(64)
        auditory = np.random.randn(32)
        system.process_sensory_input(visual, auditory)
        
        # Make a decision
        actions = ["approach", "avoid", "explore"]
        action, confidence = system.make_decision(actions)
        
        self.assertIn(action, actions)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_memory_formation_during_processing(self):
        """Test that emotionally salient events form memories."""
        system = EmbodiedCognitionSystem()
        
        # Low emotional salience - should not form memory
        visual = np.random.randn(64) * 0.1
        auditory = np.random.randn(32) * 0.1
        system.process_sensory_input(visual, auditory, threat_level=0.0, reward_signal=0.0)
        
        initial_count = system.memory.get_memory_count()
        
        # High emotional salience - should form memory
        visual = np.random.randn(64)
        auditory = np.random.randn(32)
        system.process_sensory_input(visual, auditory, threat_level=0.9, reward_signal=0.0)
        
        self.assertGreater(system.memory.get_memory_count(), initial_count)
    
    def test_state_summary(self):
        """Test state summary generation."""
        system = EmbodiedCognitionSystem()
        
        visual = np.random.randn(64)
        auditory = np.random.randn(32)
        system.process_sensory_input(visual, auditory)
        
        summary = system.get_state_summary()
        
        # Check all expected keys are present
        expected_keys = [
            'timestep', 'arousal', 'heart_rate', 'oxytocin', 'stress',
            'energy', 'homeostasis', 'attention_level', 'mood', 'motivation',
            'emotional_valence', 'emotional_arousal', 'total_memories'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)


if __name__ == '__main__':
    unittest.main()
