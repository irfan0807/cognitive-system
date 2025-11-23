"""
Live Video/Audio RAG System Demo

This demo showcases the multimodal RAG system processing live video and audio
feeds while displaying cognitive behavior in real-time.
"""

import numpy as np
import time
import logging

from cognitive_system.rag import (
    MultimodalRAGSystem,
    VideoStreamProcessor,
    AudioStreamProcessor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def simulate_live_stream_processing():
    """
    Simulate processing of live video and audio streams.
    
    This demo:
    1. Initializes the RAG system
    2. Generates synthetic video frames and audio chunks
    3. Processes them in real-time
    4. Retrieves relevant context using RAG
    5. Makes decisions based on the context
    6. Visualizes cognitive behavior
    """
    
    print("=" * 80)
    print("MULTIMODAL RAG SYSTEM - LIVE VIDEO/AUDIO DEMO")
    print("=" * 80)
    print()
    
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag_system = MultimodalRAGSystem(
        video_config={
            'frame_width': 640,
            'frame_height': 480,
            'temporal_window': 5
        },
        audio_config={
            'sample_rate': 16000,
            'chunk_size': 512
        },
        embedding_dim=128,
        visualizer_update_interval=2.0  # Update visualization every 2 seconds
    )
    rag_system.start()
    print("RAG system initialized and started!\n")
    
    # Define scenarios with different scene and sound types
    scenarios = [
        {
            'name': 'Calm Environment',
            'scene_type': 'calm',
            'sound_type': 'neutral',
            'duration': 5,
            'description': 'Processing calm visual scene with ambient audio'
        },
        {
            'name': 'Active Social Scene',
            'scene_type': 'active',
            'sound_type': 'speech',
            'duration': 5,
            'description': 'Processing active scene with speech audio'
        },
        {
            'name': 'Musical Performance',
            'scene_type': 'complex',
            'sound_type': 'music',
            'duration': 5,
            'description': 'Processing complex visual patterns with music'
        },
        {
            'name': 'Noisy Environment',
            'scene_type': 'active',
            'sound_type': 'noise',
            'duration': 5,
            'description': 'Processing active scene with noise'
        }
    ]
    
    # Define possible actions
    actions = ['observe', 'engage', 'explore', 'rest', 'respond', 'focus']
    
    # Process each scenario
    for scenario in scenarios:
        print(f"\n{'=' * 80}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'=' * 80}\n")
        
        # Simulate stream processing for this scenario
        frames_per_second = 10  # Simulate 10 FPS
        audio_chunks_per_second = 31  # ~31 chunks/sec for 512 samples at 16kHz
        
        duration = scenario['duration']
        total_frames = frames_per_second * duration
        total_chunks = audio_chunks_per_second * duration
        
        for step in range(max(total_frames, total_chunks)):
            # Generate synthetic video frame
            if step < total_frames:
                frame = rag_system.video_processor.generate_synthetic_frame(
                    scene_type=scenario['scene_type']
                )
                visual_features = rag_system.process_video_frame(frame, store_embedding=True)
            else:
                visual_features = None
            
            # Generate synthetic audio chunk
            if step < total_chunks:
                audio_chunk = rag_system.audio_processor.generate_synthetic_audio(
                    sound_type=scenario['sound_type']
                )
                audio_features = rag_system.process_audio_chunk(audio_chunk, store_embedding=True)
            else:
                audio_features = None
            
            # Create multimodal embedding if both modalities are present
            if visual_features is not None and audio_features is not None:
                # Compute attention-weighted fusion
                fusion_weights = {
                    'visual': rag_system.current_attention['visual'],
                    'audio': rag_system.current_attention['auditory']
                }
                multimodal_embedding = rag_system.create_multimodal_embedding(
                    visual_features, audio_features, fusion_weights
                )
                
                # Retrieve relevant context using RAG
                rag_context = rag_system.retrieve_context(
                    multimodal_embedding,
                    top_k=3,
                    modality_filter=None
                )
                
                # Make decision based on context
                action, confidence, _ = rag_system.make_decision(
                    multimodal_embedding,
                    actions=actions,
                    use_rag_context=True
                )
                
                decision_info = {
                    'action': action,
                    'confidence': confidence,
                    'alternatives': [a for a in actions if a != action]
                }
                
                # Update cognitive state
                rag_system.update_cognitive_state(
                    visual_features=visual_features,
                    audio_features=audio_features,
                    rag_context=rag_context
                )
                
                # Visualize cognitive behavior (only updates if interval has passed)
                visualization = rag_system.visualize_state(
                    visual_features=visual_features,
                    audio_features=audio_features,
                    rag_context=rag_context,
                    decision=decision_info
                )
                
                if visualization:
                    print(visualization)
            
            # Small delay to simulate real-time processing
            time.sleep(0.05)
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    
    stats = rag_system.get_statistics()
    
    print(f"\nSystem Performance:")
    print(f"  Total Uptime        : {stats['uptime']:.2f} seconds")
    print(f"  Frames Processed    : {stats['frames_processed']}")
    print(f"  Audio Chunks Processed: {stats['audio_chunks_processed']}")
    
    print(f"\nVector Store:")
    vs_stats = stats['vector_store_stats']
    print(f"  Total Entries       : {vs_stats['total_entries']}")
    print(f"  Embedding Dimension : {vs_stats['embedding_dim']}")
    print(f"  Modalities          : {vs_stats['modalities']}")
    
    print(f"\nFinal Cognitive State:")
    print(f"  Attention Distribution:")
    for modality, weight in stats['current_attention'].items():
        print(f"    {modality:12s}: {weight:.3f}")
    
    print(f"\n  Emotional State:")
    for param, value in stats['current_emotion'].items():
        print(f"    {param:12s}: {value:.3f}")
    
    print(f"\n  Physiological State:")
    for param, value in stats['current_physiology'].items():
        print(f"    {param:12s}: {value:.3f}")
    
    if 'visualizer_stats' in stats:
        print(f"\n  Visualizer Summary:")
        for key, value in stats['visualizer_stats'].items():
            print(f"    {key:20s}: {value}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def demonstrate_rag_retrieval():
    """
    Demonstrate RAG retrieval capabilities.
    """
    print("\n" + "=" * 80)
    print("RAG RETRIEVAL DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize a fresh RAG system
    rag_system = MultimodalRAGSystem(embedding_dim=128)
    rag_system.start()
    
    print("Building knowledge base with different types of memories...\n")
    
    # Create distinct memory types
    memory_types = [
        ('calm_scene', 'neutral', 'Calm visual scene with ambient audio'),
        ('active_scene', 'speech', 'Active social interaction with speech'),
        ('music_scene', 'music', 'Musical performance'),
        ('calm_scene', 'neutral', 'Another calm moment'),
        ('active_scene', 'noise', 'Busy environment with noise')
    ]
    
    stored_embeddings = []
    
    for scene_type, sound_type, description in memory_types:
        # Generate and process frame
        frame = rag_system.video_processor.generate_synthetic_frame(scene_type)
        visual_features = rag_system.process_video_frame(frame, store_embedding=False)
        
        # Generate and process audio
        audio = rag_system.audio_processor.generate_synthetic_audio(sound_type)
        audio_features = rag_system.process_audio_chunk(audio, store_embedding=False)
        
        # Create multimodal embedding
        multimodal = rag_system.create_multimodal_embedding(visual_features, audio_features)
        
        # Store with metadata
        metadata = {
            'modality': 'multimodal',
            'scene_type': scene_type,
            'sound_type': sound_type,
            'description': description
        }
        rag_system.vector_store.add(multimodal, metadata)
        stored_embeddings.append((multimodal, description))
        
        print(f"  Stored: {description}")
    
    print(f"\nTotal memories stored: {rag_system.vector_store.size()}")
    
    # Now test retrieval with different queries
    print("\n" + "-" * 80)
    print("Testing Retrieval with Different Queries")
    print("-" * 80)
    
    test_queries = [
        ('calm_scene', 'neutral', 'Calm scene query'),
        ('active_scene', 'speech', 'Active social scene query'),
        ('complex', 'music', 'Musical scene query')
    ]
    
    for scene_type, sound_type, query_name in test_queries:
        print(f"\nQuery: {query_name}")
        print("-" * 40)
        
        # Generate query embedding
        frame = rag_system.video_processor.generate_synthetic_frame(scene_type)
        visual_features = rag_system.video_processor.process_frame(frame)
        
        audio = rag_system.audio_processor.generate_synthetic_audio(sound_type)
        audio_features = rag_system.audio_processor.process_chunk(audio)
        
        query_embedding = rag_system.create_multimodal_embedding(visual_features, audio_features)
        
        # Retrieve similar memories
        context = rag_system.retrieve_context(query_embedding, top_k=3)
        
        print(f"Retrieved {context['retrieved_count']} memories:")
        for i, (entry, similarity) in enumerate(zip(context['entries'], context['top_similarities']), 1):
            print(f"  {i}. {entry.metadata['description']}")
            print(f"     Similarity: {similarity:.3f}")
            print(f"     Type: {entry.metadata['scene_type']} + {entry.metadata['sound_type']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        # Run the main live stream demo
        simulate_live_stream_processing()
        
        # Run the RAG retrieval demonstration
        demonstrate_rag_retrieval()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        raise
