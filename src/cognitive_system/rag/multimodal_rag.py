"""
Multimodal RAG System

Main RAG system that integrates video/audio stream processing, vector storage,
and retrieval-augmented generation for the cognitive system.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import time

from .vector_store import VectorStore
from .stream_processor import VideoStreamProcessor, AudioStreamProcessor
from .cognitive_visualizer import CognitiveVisualizer


class MultimodalRAGSystem:
    """
    Complete RAG system for live video and audio streams.
    
    This system:
    1. Processes live video and audio streams
    2. Extracts multimodal embeddings
    3. Stores embeddings in a vector database
    4. Retrieves relevant context for decision-making
    5. Integrates with the cognitive system
    6. Visualizes cognitive behavior in real-time
    """
    
    def __init__(self, 
                 video_config: Optional[Dict[str, Any]] = None,
                 audio_config: Optional[Dict[str, Any]] = None,
                 embedding_dim: int = 128,
                 visualizer_update_interval: float = 0.5):
        """
        Initialize the multimodal RAG system.
        
        Args:
            video_config: Configuration for video stream processor
            audio_config: Configuration for audio stream processor
            embedding_dim: Dimension for embeddings
            visualizer_update_interval: Update interval for visualizer (seconds)
        """
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        video_config = video_config or {}
        audio_config = audio_config or {}
        
        self.video_processor = VideoStreamProcessor(
            frame_width=video_config.get('frame_width', 640),
            frame_height=video_config.get('frame_height', 480),
            feature_dim=embedding_dim,
            temporal_window=video_config.get('temporal_window', 5)
        )
        
        self.audio_processor = AudioStreamProcessor(
            sample_rate=audio_config.get('sample_rate', 16000),
            chunk_size=audio_config.get('chunk_size', 512),
            feature_dim=embedding_dim
        )
        
        self.vector_store = VectorStore(embedding_dim=embedding_dim)
        
        self.visualizer = CognitiveVisualizer(
            update_interval=visualizer_update_interval
        )
        
        # System state
        self.is_active = False
        self.start_time = None
        self.frame_count = 0
        self.chunk_count = 0
        
        # Cognitive integration state
        self.cognitive_system = None
        self.current_attention = {'visual': 0.5, 'auditory': 0.5, 'memory': 0.3, 'internal': 0.2}
        self.current_emotion = {'valence': 0.5, 'arousal': 0.3, 'mood': 0.6}
        self.current_physiology = {'heart_rate': 70.0, 'arousal': 0.3, 'stress': 0.2, 'energy': 0.8}
        
        self.logger.info(f"MultimodalRAGSystem initialized with embedding_dim={embedding_dim}")
    
    def attach_cognitive_system(self, cognitive_system):
        """
        Attach a cognitive system for integration.
        
        Args:
            cognitive_system: The cognitive system instance to integrate with
        """
        self.cognitive_system = cognitive_system
        self.logger.info("Cognitive system attached to RAG system")
    
    def start(self):
        """Start the RAG system."""
        self.is_active = True
        self.start_time = time.time()
        self.logger.info("RAG system started")
    
    def stop(self):
        """Stop the RAG system."""
        self.is_active = False
        self.logger.info("RAG system stopped")
    
    def process_video_frame(self, frame: np.ndarray, 
                           store_embedding: bool = True) -> np.ndarray:
        """
        Process a video frame and optionally store its embedding.
        
        Args:
            frame: Video frame as numpy array
            store_embedding: Whether to store the embedding in vector store
            
        Returns:
            Extracted feature vector
        """
        if not self.is_active:
            self.logger.warning("RAG system not active, starting...")
            self.start()
        
        # Extract features
        features = self.video_processor.process_frame(frame)
        
        # Store in vector database
        if store_embedding:
            metadata = {
                'modality': 'visual',
                'frame_number': self.frame_count,
                'processing_time': time.time(),
                'scene_complexity': float(np.std(features))
            }
            self.vector_store.add(features, metadata)
        
        self.frame_count += 1
        
        return features
    
    def process_audio_chunk(self, audio_chunk: np.ndarray,
                           store_embedding: bool = True) -> np.ndarray:
        """
        Process an audio chunk and optionally store its embedding.
        
        Args:
            audio_chunk: Audio samples as numpy array
            store_embedding: Whether to store the embedding in vector store
            
        Returns:
            Extracted feature vector
        """
        if not self.is_active:
            self.logger.warning("RAG system not active, starting...")
            self.start()
        
        # Extract features
        features = self.audio_processor.process_chunk(audio_chunk)
        
        # Store in vector database
        if store_embedding:
            metadata = {
                'modality': 'auditory',
                'chunk_number': self.chunk_count,
                'processing_time': time.time(),
                'audio_intensity': float(np.mean(np.abs(audio_chunk)))
            }
            self.vector_store.add(features, metadata)
        
        self.chunk_count += 1
        
        return features
    
    def create_multimodal_embedding(self, 
                                   visual_features: np.ndarray,
                                   audio_features: np.ndarray,
                                   fusion_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Create a multimodal embedding by fusing visual and audio features.
        
        Args:
            visual_features: Visual feature vector
            audio_features: Audio feature vector
            fusion_weights: Weights for fusion (default: equal weighting)
            
        Returns:
            Fused multimodal embedding
        """
        if fusion_weights is None:
            fusion_weights = {'visual': 0.5, 'audio': 0.5}
        
        # Normalize weights
        total_weight = fusion_weights['visual'] + fusion_weights['audio']
        if total_weight > 0:
            fusion_weights = {k: v / total_weight for k, v in fusion_weights.items()}
        
        # Weighted fusion
        multimodal = (fusion_weights['visual'] * visual_features + 
                     fusion_weights['audio'] * audio_features)
        
        # Normalize
        norm = np.linalg.norm(multimodal)
        if norm > 0:
            multimodal = multimodal / norm
        
        return multimodal
    
    def retrieve_context(self, 
                        query_embedding: np.ndarray,
                        top_k: int = 5,
                        modality_filter: Optional[str] = None,
                        time_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant context from the vector store.
        
        Args:
            query_embedding: Query vector for retrieval
            top_k: Number of results to retrieve
            modality_filter: Optional modality filter
            time_window: Optional time window for retrieval
            
        Returns:
            Dictionary with retrieval results and metadata
        """
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            modality_filter=modality_filter,
            time_window=time_window
        )
        
        # Extract information
        retrieved_entries = []
        similarities = []
        memory_types = set()
        
        for entry, similarity in results:
            retrieved_entries.append(entry)
            similarities.append(similarity)
            memory_types.add(entry.metadata.get('modality', 'unknown'))
        
        # Compute context relevance (average similarity)
        context_relevance = np.mean(similarities) if similarities else 0.0
        
        context = {
            'retrieved_count': len(results),
            'entries': retrieved_entries,
            'top_similarities': similarities,
            'context_relevance': float(context_relevance),
            'memory_types': list(memory_types)
        }
        
        return context
    
    def update_cognitive_state(self, 
                               visual_features: Optional[np.ndarray] = None,
                               audio_features: Optional[np.ndarray] = None,
                               rag_context: Optional[Dict[str, Any]] = None):
        """
        Update the cognitive state based on current input and context.
        
        Args:
            visual_features: Current visual features
            audio_features: Current audio features
            rag_context: Retrieved context from RAG
        """
        # Update attention based on feature intensity
        if visual_features is not None:
            visual_intensity = float(np.abs(visual_features).mean())
            self.current_attention['visual'] = 0.3 + 0.7 * visual_intensity
        
        if audio_features is not None:
            audio_intensity = float(np.abs(audio_features).mean())
            self.current_attention['auditory'] = 0.3 + 0.7 * audio_intensity
        
        # Update attention to memory based on context relevance
        if rag_context and 'context_relevance' in rag_context:
            self.current_attention['memory'] = rag_context['context_relevance']
        
        # Normalize attention
        total_attention = sum(self.current_attention.values())
        if total_attention > 0:
            self.current_attention = {k: v / total_attention for k, v in self.current_attention.items()}
        
        # Update arousal based on multimodal intensity
        if visual_features is not None and audio_features is not None:
            combined_intensity = (np.abs(visual_features).mean() + np.abs(audio_features).mean()) / 2
            self.current_physiology['arousal'] = 0.2 + 0.6 * float(combined_intensity)
            
            # Heart rate increases with arousal
            base_hr = 60.0
            self.current_physiology['heart_rate'] = base_hr + 40 * self.current_physiology['arousal']
        
        # Update emotion based on context and features
        if rag_context and 'context_relevance' in rag_context:
            # High context relevance can increase positive valence (familiarity)
            self.current_emotion['valence'] = 0.5 + 0.3 * rag_context['context_relevance']
        
        self.current_emotion['arousal'] = self.current_physiology['arousal']
    
    def visualize_state(self, 
                       visual_features: Optional[np.ndarray] = None,
                       audio_features: Optional[np.ndarray] = None,
                       rag_context: Optional[Dict[str, Any]] = None,
                       decision: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate a visualization of the current cognitive state.
        
        Args:
            visual_features: Current visual features
            audio_features: Current audio features
            rag_context: Retrieved context
            decision: Decision information
            
        Returns:
            Visualization string if update is due, None otherwise
        """
        if not self.visualizer.should_update():
            return None
        
        # Prepare state dictionary
        state = {
            'attention': self.current_attention,
            'physiology': self.current_physiology,
            'emotion': self.current_emotion
        }
        
        if decision:
            state['decision'] = decision
        
        # Memory usage
        state['memory_usage'] = {
            'retrievals': len(rag_context.get('entries', [])) if rag_context else 0,
            'stored': self.vector_store.size(),
            'total_memories': self.vector_store.size()
        }
        
        # Stream info
        stream_info = {
            'video_frame_count': self.frame_count,
            'audio_chunk_count': self.chunk_count
        }
        
        if visual_features is not None:
            stream_info['video_features'] = visual_features
        
        if audio_features is not None:
            stream_info['audio_features'] = audio_features
        
        # Generate visualization
        visualization = self.visualizer.visualize_cognitive_state(
            state=state,
            stream_info=stream_info,
            rag_context=rag_context
        )
        
        return visualization
    
    def make_decision(self, 
                     current_embedding: np.ndarray,
                     actions: List[str],
                     use_rag_context: bool = True) -> Tuple[str, float, Dict[str, Any]]:
        """
        Make a decision based on current state and optionally RAG context.
        
        Args:
            current_embedding: Current multimodal embedding
            actions: List of possible actions
            use_rag_context: Whether to use RAG context for decision
            
        Returns:
            Tuple of (selected_action, confidence, context_dict)
        """
        context_dict = {}
        
        # Retrieve relevant context if enabled
        if use_rag_context:
            context_dict = self.retrieve_context(current_embedding, top_k=3)
        
        # Compute action scores based on current state and context
        action_scores = []
        for action in actions:
            # Base score from random exploration
            score = np.random.rand() * 0.3
            
            # Add score based on arousal (prefer active actions when aroused)
            if 'explore' in action.lower() or 'active' in action.lower():
                score += self.current_physiology['arousal'] * 0.3
            
            # Add score based on context relevance
            if use_rag_context and 'context_relevance' in context_dict:
                score += context_dict['context_relevance'] * 0.4
            
            action_scores.append(score)
        
        # Select action with highest score
        best_idx = np.argmax(action_scores)
        selected_action = actions[best_idx]
        confidence = float(action_scores[best_idx])
        
        # Normalize confidence to 0-1
        if max(action_scores) > 0:
            confidence = confidence / max(action_scores)
        
        decision = {
            'action': selected_action,
            'confidence': confidence,
            'alternatives': [a for i, a in enumerate(actions) if i != best_idx]
        }
        
        return selected_action, confidence, context_dict
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            'is_active': self.is_active,
            'uptime': time.time() - self.start_time if self.start_time else 0.0,
            'frames_processed': self.frame_count,
            'audio_chunks_processed': self.chunk_count,
            'vector_store_stats': self.vector_store.get_stats(),
            'current_attention': self.current_attention,
            'current_emotion': self.current_emotion,
            'current_physiology': self.current_physiology
        }
        
        # Add visualizer statistics
        vis_stats = self.visualizer.get_summary_statistics()
        stats['visualizer_stats'] = vis_stats
        
        return stats
    
    def reset(self):
        """Reset the RAG system to initial state."""
        self.video_processor.reset()
        self.audio_processor.reset()
        self.vector_store.clear()
        
        self.frame_count = 0
        self.chunk_count = 0
        self.start_time = time.time() if self.is_active else None
        
        # Reset cognitive state
        self.current_attention = {'visual': 0.5, 'auditory': 0.5, 'memory': 0.3, 'internal': 0.2}
        self.current_emotion = {'valence': 0.5, 'arousal': 0.3, 'mood': 0.6}
        self.current_physiology = {'heart_rate': 70.0, 'arousal': 0.3, 'stress': 0.2, 'energy': 0.8}
        
        self.logger.info("RAG system reset")
