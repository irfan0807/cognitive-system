"""
Tests for the RAG (Retrieval-Augmented Generation) System

This test suite validates the functionality of the multimodal RAG system
including vector storage, stream processing, and cognitive visualization.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_system.rag import (
    MultimodalRAGSystem,
    VectorStore,
    VideoStreamProcessor,
    AudioStreamProcessor,
    CognitiveVisualizer
)


def test_vector_store():
    """Test the vector store functionality."""
    print("\nTesting Vector Store...")
    
    # Initialize vector store
    store = VectorStore(embedding_dim=64)
    assert store.size() == 0, "Initial store should be empty"
    
    # Add some vectors
    embeddings = []
    for i in range(5):
        embedding = np.random.randn(64)
        metadata = {'modality': 'visual' if i % 2 == 0 else 'auditory', 'index': i}
        entry_id = store.add(embedding, metadata)
        embeddings.append(embedding)
        assert entry_id is not None, "Entry ID should be returned"
    
    assert store.size() == 5, "Store should contain 5 entries"
    
    # Test search
    query = embeddings[0]
    results = store.search(query, top_k=3)
    assert len(results) == 3, "Should return top 3 results"
    assert results[0][1] > 0.9, "First result should have high similarity"
    
    # Test modality filter
    visual_results = store.search(query, top_k=5, modality_filter='visual')
    assert all(r[0].metadata['modality'] == 'visual' for r in visual_results), \
        "Should only return visual results"
    
    # Test get_recent
    recent = store.get_recent(n=2)
    assert len(recent) == 2, "Should return 2 recent entries"
    
    # Test clear
    store.clear(modality='visual')
    assert store.size() == 2, "Should have 2 entries left (auditory)"
    
    print("✓ Vector Store tests passed")


def test_video_stream_processor():
    """Test the video stream processor."""
    print("\nTesting Video Stream Processor...")
    
    # Initialize processor
    processor = VideoStreamProcessor(
        frame_width=640,
        frame_height=480,
        feature_dim=128,
        temporal_window=3
    )
    
    # Generate and process a synthetic frame
    frame = processor.generate_synthetic_frame(scene_type='calm')
    assert frame.shape == (480, 640, 3), "Frame should have correct shape"
    
    features = processor.process_frame(frame)
    assert features.shape == (128,), "Features should have correct dimension"
    assert processor.frame_count == 1, "Frame count should be 1"
    
    # Process multiple frames to test temporal features
    for _ in range(5):
        frame = processor.generate_synthetic_frame(scene_type='active')
        features = processor.process_frame(frame)
        assert features.shape == (128,), "Features should have correct dimension"
    
    assert processor.frame_count == 6, "Frame count should be 6"
    assert len(processor.frame_buffer) == 3, "Buffer should respect temporal window"
    
    # Test reset
    processor.reset()
    assert processor.frame_count == 0, "Frame count should be reset"
    assert len(processor.frame_buffer) == 0, "Buffer should be cleared"
    
    print("✓ Video Stream Processor tests passed")


def test_audio_stream_processor():
    """Test the audio stream processor."""
    print("\nTesting Audio Stream Processor...")
    
    # Initialize processor
    processor = AudioStreamProcessor(
        sample_rate=16000,
        chunk_size=512,
        feature_dim=128
    )
    
    # Generate and process a synthetic audio chunk
    audio = processor.generate_synthetic_audio(sound_type='speech')
    assert len(audio) == 512, "Audio should have correct length"
    
    features = processor.process_chunk(audio)
    assert features.shape == (128,), "Features should have correct dimension"
    assert processor.chunk_count == 1, "Chunk count should be 1"
    
    # Process multiple chunks
    for sound_type in ['music', 'noise', 'neutral']:
        audio = processor.generate_synthetic_audio(sound_type=sound_type)
        features = processor.process_chunk(audio)
        assert features.shape == (128,), "Features should have correct dimension"
    
    assert processor.chunk_count == 4, "Chunk count should be 4"
    
    # Test reset
    processor.reset()
    assert processor.chunk_count == 0, "Chunk count should be reset"
    assert len(processor.audio_buffer) == 0, "Buffer should be cleared"
    
    print("✓ Audio Stream Processor tests passed")


def test_cognitive_visualizer():
    """Test the cognitive visualizer."""
    print("\nTesting Cognitive Visualizer...")
    
    # Initialize visualizer
    visualizer = CognitiveVisualizer(update_interval=0.1)
    
    # Test should_update (should be True initially)
    assert visualizer.should_update() == True, "Should update initially"
    
    # Test again immediately (should be False)
    assert visualizer.should_update() == False, "Should not update immediately"
    
    # Wait and test again
    time.sleep(0.15)
    assert visualizer.should_update() == True, "Should update after interval"
    
    # Test visualization generation
    state = {
        'attention': {'visual': 0.5, 'auditory': 0.3, 'memory': 0.2},
        'physiology': {'heart_rate': 75.0, 'arousal': 0.6, 'stress': 0.3},
        'emotion': {'valence': 0.7, 'arousal': 0.6, 'mood': 0.8},
        'decision': {'action': 'explore', 'confidence': 0.85}
    }
    
    stream_info = {
        'video_frame_count': 10,
        'audio_chunk_count': 20,
        'video_features': np.random.randn(128),
        'audio_features': np.random.randn(128)
    }
    
    rag_context = {
        'retrieved_count': 3,
        'top_similarities': [0.9, 0.8, 0.7],
        'context_relevance': 0.8,
        'memory_types': ['visual', 'auditory']
    }
    
    visualization = visualizer.visualize_cognitive_state(
        state=state,
        stream_info=stream_info,
        rag_context=rag_context
    )
    
    assert isinstance(visualization, str), "Visualization should be a string"
    assert "COGNITIVE BEHAVIOR" in visualization, "Should contain header"
    assert "ATTENTION" in visualization, "Should contain attention section"
    assert "PHYSIOLOGICAL" in visualization, "Should contain physiology section"
    
    # Test summary statistics
    stats = visualizer.get_summary_statistics()
    assert isinstance(stats, dict), "Statistics should be a dictionary"
    
    print("✓ Cognitive Visualizer tests passed")


def test_multimodal_rag_system():
    """Test the complete multimodal RAG system."""
    print("\nTesting Multimodal RAG System...")
    
    # Initialize RAG system
    rag = MultimodalRAGSystem(
        embedding_dim=128,
        visualizer_update_interval=0.1
    )
    
    # Test start/stop
    rag.start()
    assert rag.is_active == True, "System should be active"
    assert rag.start_time is not None, "Start time should be set"
    
    # Process video frame
    frame = rag.video_processor.generate_synthetic_frame('calm')
    visual_features = rag.process_video_frame(frame, store_embedding=True)
    assert visual_features.shape == (128,), "Visual features should have correct dimension"
    assert rag.frame_count == 1, "Frame count should be 1"
    assert rag.vector_store.size() == 1, "Vector store should contain 1 entry"
    
    # Process audio chunk
    audio = rag.audio_processor.generate_synthetic_audio('speech')
    audio_features = rag.process_audio_chunk(audio, store_embedding=True)
    assert audio_features.shape == (128,), "Audio features should have correct dimension"
    assert rag.chunk_count == 1, "Chunk count should be 1"
    assert rag.vector_store.size() == 2, "Vector store should contain 2 entries"
    
    # Test multimodal embedding fusion
    multimodal = rag.create_multimodal_embedding(
        visual_features,
        audio_features,
        fusion_weights={'visual': 0.6, 'audio': 0.4}
    )
    assert multimodal.shape == (128,), "Multimodal embedding should have correct dimension"
    
    # Test context retrieval
    context = rag.retrieve_context(multimodal, top_k=2)
    assert context['retrieved_count'] == 2, "Should retrieve 2 entries"
    assert len(context['top_similarities']) == 2, "Should have 2 similarities"
    assert 'context_relevance' in context, "Should have context relevance"
    
    # Test cognitive state update
    rag.update_cognitive_state(
        visual_features=visual_features,
        audio_features=audio_features,
        rag_context=context
    )
    assert 'visual' in rag.current_attention, "Should have visual attention"
    assert 'arousal' in rag.current_physiology, "Should have arousal"
    
    # Test decision making
    actions = ['observe', 'engage', 'explore', 'rest']
    action, confidence, decision_context = rag.make_decision(
        multimodal,
        actions=actions,
        use_rag_context=True
    )
    assert action in actions, "Should select a valid action"
    assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"
    
    # Test visualization
    time.sleep(0.15)  # Wait for update interval
    viz = rag.visualize_state(
        visual_features=visual_features,
        audio_features=audio_features,
        rag_context=context,
        decision={'action': action, 'confidence': confidence}
    )
    assert viz is not None, "Should generate visualization"
    assert isinstance(viz, str), "Visualization should be a string"
    
    # Test statistics
    stats = rag.get_statistics()
    assert stats['is_active'] == True, "Should be active"
    assert stats['frames_processed'] == 1, "Should have processed 1 frame"
    assert stats['audio_chunks_processed'] == 1, "Should have processed 1 chunk"
    assert 'vector_store_stats' in stats, "Should have vector store stats"
    
    # Test reset
    rag.reset()
    assert rag.frame_count == 0, "Frame count should be reset"
    assert rag.chunk_count == 0, "Chunk count should be reset"
    assert rag.vector_store.size() == 0, "Vector store should be empty"
    
    # Test stop
    rag.stop()
    assert rag.is_active == False, "System should be stopped"
    
    print("✓ Multimodal RAG System tests passed")


def test_integration_scenario():
    """Test a complete integration scenario."""
    print("\nTesting Integration Scenario...")
    
    # Initialize RAG system
    rag = MultimodalRAGSystem(embedding_dim=64, visualizer_update_interval=0.05)
    rag.start()
    
    # Simulate processing a short sequence
    num_steps = 10
    actions = ['observe', 'engage', 'explore']
    
    for i in range(num_steps):
        # Process video and audio
        scene_type = 'calm' if i < 5 else 'active'
        sound_type = 'neutral' if i < 5 else 'speech'
        
        frame = rag.video_processor.generate_synthetic_frame(scene_type)
        visual_features = rag.process_video_frame(frame)
        
        audio = rag.audio_processor.generate_synthetic_audio(sound_type)
        audio_features = rag.process_audio_chunk(audio)
        
        # Create multimodal embedding
        multimodal = rag.create_multimodal_embedding(visual_features, audio_features)
        
        # Retrieve context and make decision
        if i > 0:  # Need some history for retrieval
            context = rag.retrieve_context(multimodal, top_k=min(3, i))
            action, confidence, _ = rag.make_decision(multimodal, actions)
            
            rag.update_cognitive_state(visual_features, audio_features, context)
    
    # Verify final state
    assert rag.frame_count == num_steps, f"Should have processed {num_steps} frames"
    assert rag.chunk_count == num_steps, f"Should have processed {num_steps} chunks"
    assert rag.vector_store.size() == num_steps * 2, "Should have stored all embeddings"
    
    stats = rag.get_statistics()
    assert stats['uptime'] > 0, "Should have positive uptime"
    
    print("✓ Integration Scenario test passed")


def run_all_tests():
    """Run all RAG system tests."""
    print("=" * 60)
    print("Running RAG System Tests")
    print("=" * 60)
    
    tests = [
        test_vector_store,
        test_video_stream_processor,
        test_audio_stream_processor,
        test_cognitive_visualizer,
        test_multimodal_rag_system,
        test_integration_scenario
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
