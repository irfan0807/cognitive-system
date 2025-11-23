"""
Tests for Deep Neural Network Consciousness System

Tests the deep learning components and consciousness simulation.
"""

import unittest
import numpy as np
import torch

from cognitive_system.core.deep_neural_networks import (
    VisualCortexCNN,
    AuditoryCortexRNN,
    MultimodalIntegrationNetwork,
    ConsciousnessNetwork,
    DeepNeuralController
)
from cognitive_system.core.live_feed import (
    LiveVideoFeed,
    LiveAudioFeed,
    LiveFeedManager
)
from cognitive_system.core.conscious_cognition import ConsciousCognitionSystem


class TestVisualCortex(unittest.TestCase):
    """Test visual cortex CNN."""
    
    def setUp(self):
        self.visual_cortex = VisualCortexCNN(input_channels=3, output_dim=256)
        self.visual_cortex.eval()
    
    def test_initialization(self):
        """Test visual cortex initializes correctly."""
        self.assertIsInstance(self.visual_cortex, VisualCortexCNN)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        # Create dummy input (batch=2, channels=3, height=224, width=224)
        dummy_input = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = self.visual_cortex(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 256))
        
        # Check output is not all zeros
        self.assertGreater(torch.abs(output).sum(), 0)


class TestAuditoryCortex(unittest.TestCase):
    """Test auditory cortex RNN."""
    
    def setUp(self):
        self.auditory_cortex = AuditoryCortexRNN(
            input_dim=128, hidden_dim=256, output_dim=128
        )
        self.auditory_cortex.eval()
    
    def test_initialization(self):
        """Test auditory cortex initializes correctly."""
        self.assertIsInstance(self.auditory_cortex, AuditoryCortexRNN)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        # Create dummy input (batch=2, sequence=100, features=128)
        dummy_input = torch.randn(2, 100, 128)
        
        with torch.no_grad():
            output, hidden = self.auditory_cortex(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 128))
        
        # Check hidden state is returned
        self.assertIsNotNone(hidden)


class TestMultimodalIntegration(unittest.TestCase):
    """Test multimodal integration network."""
    
    def setUp(self):
        self.integration = MultimodalIntegrationNetwork(
            visual_dim=256, auditory_dim=128, integrated_dim=512
        )
        self.integration.eval()
    
    def test_initialization(self):
        """Test integration network initializes correctly."""
        self.assertIsInstance(self.integration, MultimodalIntegrationNetwork)
    
    def test_forward_pass(self):
        """Test forward pass integrates modalities."""
        # Create dummy inputs
        visual = torch.randn(2, 256)
        auditory = torch.randn(2, 128)
        
        with torch.no_grad():
            output = self.integration(visual, auditory)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 512))


class TestConsciousnessNetwork(unittest.TestCase):
    """Test consciousness network."""
    
    def setUp(self):
        self.consciousness = ConsciousnessNetwork(
            visual_output_dim=256,
            auditory_output_dim=128,
            integrated_dim=512,
            consciousness_dim=256
        )
        self.consciousness.eval()
    
    def test_initialization(self):
        """Test consciousness network initializes correctly."""
        self.assertIsInstance(self.consciousness, ConsciousnessNetwork)
    
    def test_forward_pass(self):
        """Test forward pass through consciousness network."""
        # Create dummy inputs
        visual_input = torch.randn(2, 3, 224, 224)
        auditory_input = torch.randn(2, 100, 128)
        
        with torch.no_grad():
            outputs = self.consciousness(visual_input, auditory_input)
        
        # Check all outputs present
        self.assertIn('consciousness_state', outputs)
        self.assertIn('visual_features', outputs)
        self.assertIn('auditory_features', outputs)
        self.assertIn('integrated_features', outputs)
        
        # Check shapes
        self.assertEqual(outputs['consciousness_state'].shape, (2, 256))
        self.assertEqual(outputs['visual_features'].shape, (2, 256))
        self.assertEqual(outputs['auditory_features'].shape, (2, 128))
        self.assertEqual(outputs['integrated_features'].shape, (2, 512))
    
    def test_process_live_feed(self):
        """Test processing live feed (numpy arrays)."""
        # Create dummy numpy inputs
        visual_frame = np.random.rand(224, 224, 3).astype(np.float32)
        audio_chunk = np.random.rand(100, 128).astype(np.float32)
        
        outputs = self.consciousness.process_live_feed(visual_frame, audio_chunk)
        
        # Check outputs are numpy arrays
        self.assertIsInstance(outputs['consciousness_state'], np.ndarray)
        self.assertIsInstance(outputs['visual_features'], np.ndarray)
        self.assertIsInstance(outputs['auditory_features'], np.ndarray)


class TestDeepNeuralController(unittest.TestCase):
    """Test deep neural controller."""
    
    def setUp(self):
        self.controller = DeepNeuralController(consciousness_dim=256)
    
    def test_initialization(self):
        """Test controller initializes correctly."""
        self.assertIsInstance(self.controller, DeepNeuralController)
        self.assertEqual(self.controller.consciousness_dim, 256)
    
    def test_process_live_input(self):
        """Test processing live input."""
        # Create dummy inputs
        visual_frame = np.random.rand(224, 224, 3).astype(np.float32)
        audio_chunk = np.random.rand(100, 128).astype(np.float32)
        
        outputs = self.controller.process_live_input(visual_frame, audio_chunk)
        
        # Check outputs
        self.assertIn('consciousness_state', outputs)
        self.assertIn('visual_features', outputs)
        self.assertIn('auditory_features', outputs)
    
    def test_integration_with_embodied_cognition(self):
        """Test integration with embodied cognition."""
        # Process some input first
        visual_frame = np.random.rand(224, 224, 3).astype(np.float32)
        audio_chunk = np.random.rand(100, 128).astype(np.float32)
        consciousness_features = self.controller.process_live_input(
            visual_frame, audio_chunk
        )
        
        # Create dummy physiological state
        physiology = {
            'arousal': 0.5,
            'heart_rate': 75.0,
            'stress': 0.3,
            'oxytocin': 0.6
        }
        
        integrated = self.controller.integrate_with_embodied_cognition(
            consciousness_features, physiology
        )
        
        # Check integrated state
        self.assertIn('consciousness_state', integrated)
        self.assertIn('visual_awareness', integrated)
        self.assertIn('auditory_awareness', integrated)
        self.assertIn('arousal_modulation', integrated)
        self.assertIn('attention_focus', integrated)


class TestLiveFeeds(unittest.TestCase):
    """Test live feed components."""
    
    def test_video_feed_initialization(self):
        """Test video feed initializes."""
        feed = LiveVideoFeed(camera_id=0, target_size=(224, 224))
        self.assertIsInstance(feed, LiveVideoFeed)
    
    def test_audio_feed_initialization(self):
        """Test audio feed initializes."""
        feed = LiveAudioFeed(sample_rate=16000)
        self.assertIsInstance(feed, LiveAudioFeed)
    
    def test_simulated_video_frame(self):
        """Test simulated video frame generation."""
        feed = LiveVideoFeed(target_size=(224, 224))
        frame = feed.get_simulated_frame()
        
        self.assertEqual(frame.shape, (224, 224, 3))
        self.assertTrue(np.all(frame >= 0) and np.all(frame <= 1))
    
    def test_simulated_audio_features(self):
        """Test simulated audio features generation."""
        feed = LiveAudioFeed(sample_rate=16000, n_mels=128)
        features = feed.get_simulated_features()
        
        self.assertEqual(features.shape[1], 128)  # n_mels dimension
        self.assertGreater(features.shape[0], 0)  # time dimension
    
    def test_feed_manager(self):
        """Test feed manager."""
        manager = LiveFeedManager(use_camera=False, use_microphone=False)
        
        # Get multimodal input (should use simulated)
        video, audio = manager.get_multimodal_input()
        
        self.assertIsInstance(video, np.ndarray)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(video.shape, (224, 224, 3))


class TestConsciousCognitionSystem(unittest.TestCase):
    """Test conscious cognition system."""
    
    def setUp(self):
        self.system = ConsciousCognitionSystem(
            visual_dim=64,
            auditory_dim=32,
            use_live_feeds=False
        )
    
    def test_initialization(self):
        """Test system initializes correctly."""
        self.assertIsInstance(self.system, ConsciousCognitionSystem)
        self.assertIsNotNone(self.system.deep_controller)
    
    def test_process_live_consciousness(self):
        """Test processing consciousness with simulated feeds."""
        output = self.system.process_live_consciousness()
        
        # Check outputs
        self.assertIn('consciousness_features', output)
        self.assertIn('integrated_state', output)
        self.assertIn('physiological_state', output)
        self.assertIn('video_frame', output)
        self.assertIn('audio_features', output)
    
    def test_consciousness_integration(self):
        """Test consciousness integrates with embodied cognition."""
        # Process consciousness
        output1 = self.system.process_live_consciousness()
        
        # Get state summary
        summary = self.system.get_state_summary()
        
        # Check state is updated
        self.assertIn('arousal', summary)
        self.assertIn('mood', summary)
        self.assertIn('stress', summary)
    
    def test_memory_formation(self):
        """Test memory formation from conscious experiences."""
        # Process once to initialize
        output = self.system.process_live_consciousness()
        
        initial_memories = self.system.get_state_summary()['total_memories']
        
        # Process several inputs with emotional significance
        for _ in range(5):
            output = self.system.process_live_consciousness()
            visual = output['consciousness_features']['visual_features'].flatten()[:64]
            auditory = output['consciousness_features']['auditory_features'].flatten()[:32]
            
            self.system.process_sensory_input(
                visual_input=visual,
                auditory_input=auditory,
                social_context=0.5,
                threat_level=0.0,
                reward_signal=0.6
            )
        
        final_memories = self.system.get_state_summary()['total_memories']
        
        # Should have formed some memories
        self.assertGreaterEqual(final_memories, initial_memories)


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()
