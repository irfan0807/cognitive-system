"""
Multimodal Memory System

Integrates emotion, visual, and auditory stimuli into coherent memories.
Implements memory formation, storage, and retrieval.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MultimodalMemory:
    """
    Represents a single multimodal memory integrating multiple sensory modalities.
    """
    timestamp: datetime
    emotional_valence: float  # -1.0 (negative) to 1.0 (positive)
    emotional_arousal: float  # 0.0 (calm) to 1.0 (excited)
    visual_features: np.ndarray  # Visual stimulus encoding
    auditory_features: np.ndarray  # Auditory stimulus encoding
    physiological_state: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, any] = field(default_factory=dict)
    strength: float = 1.0  # Memory strength (0-1), decays over time
    
    def get_embedding(self) -> np.ndarray:
        """
        Get a combined embedding of this memory.
        
        Returns:
            Numpy array representing the memory embedding
        """
        emotion_vec = np.array([self.emotional_valence, self.emotional_arousal])
        return np.concatenate([
            emotion_vec,
            self.visual_features,
            self.auditory_features
        ])


class MemoryConsolidation:
    """
    Handles memory consolidation - the process of strengthening important memories.
    """
    
    def __init__(self, emotional_weight: float = 0.3, arousal_weight: float = 0.3):
        """
        Initialize consolidation parameters.
        
        Args:
            emotional_weight: How much emotional valence affects consolidation
            arousal_weight: How much arousal affects consolidation
        """
        self.emotional_weight = emotional_weight
        self.arousal_weight = arousal_weight
    
    def compute_consolidation_strength(self, memory: MultimodalMemory) -> float:
        """
        Compute how strongly a memory should be consolidated.
        
        Args:
            memory: The memory to evaluate
            
        Returns:
            Consolidation strength (0-1)
        """
        # Emotional memories are consolidated more strongly
        emotional_factor = abs(memory.emotional_valence) * self.emotional_weight
        arousal_factor = memory.emotional_arousal * self.arousal_weight
        
        # Base consolidation from repetition/importance
        base_strength = 0.4
        
        consolidation = base_strength + emotional_factor + arousal_factor
        return np.clip(consolidation, 0.0, 1.0)


class MultimodalMemorySystem:
    """
    Complete multimodal memory system integrating emotion, visual, and auditory memories.
    """
    
    def __init__(
        self,
        visual_dim: int = 64,
        auditory_dim: int = 32,
        max_memories: int = 1000,
        decay_rate: float = 0.01
    ):
        """
        Initialize the memory system.
        
        Args:
            visual_dim: Dimensionality of visual feature encoding
            auditory_dim: Dimensionality of auditory feature encoding
            max_memories: Maximum number of memories to retain
            decay_rate: Rate at which memories decay over time
        """
        self.visual_dim = visual_dim
        self.auditory_dim = auditory_dim
        self.max_memories = max_memories
        self.decay_rate = decay_rate
        
        self.memories: List[MultimodalMemory] = []
        self.consolidator = MemoryConsolidation()
    
    def encode_visual(self, visual_input: np.ndarray) -> np.ndarray:
        """
        Encode visual input into feature representation.
        
        Args:
            visual_input: Raw visual input
            
        Returns:
            Visual feature vector
        """
        # Simplified encoding - in a real system this would use a CNN
        if len(visual_input) >= self.visual_dim:
            return visual_input[:self.visual_dim]
        else:
            # Pad if needed
            padded = np.zeros(self.visual_dim)
            padded[:len(visual_input)] = visual_input
            return padded
    
    def encode_auditory(self, auditory_input: np.ndarray) -> np.ndarray:
        """
        Encode auditory input into feature representation.
        
        Args:
            auditory_input: Raw auditory input
            
        Returns:
            Auditory feature vector
        """
        # Simplified encoding - in a real system this would use spectral features
        if len(auditory_input) >= self.auditory_dim:
            return auditory_input[:self.auditory_dim]
        else:
            # Pad if needed
            padded = np.zeros(self.auditory_dim)
            padded[:len(auditory_input)] = auditory_input
            return padded
    
    def form_memory(
        self,
        visual_input: np.ndarray,
        auditory_input: np.ndarray,
        emotional_valence: float,
        emotional_arousal: float,
        physiological_state: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, any]] = None
    ) -> MultimodalMemory:
        """
        Form a new multimodal memory from current sensory inputs.
        
        Args:
            visual_input: Visual stimulus
            auditory_input: Auditory stimulus
            emotional_valence: Emotional valence (-1 to 1)
            emotional_arousal: Emotional arousal (0 to 1)
            physiological_state: Current physiological state
            context: Additional context information
            
        Returns:
            Newly formed memory
        """
        memory = MultimodalMemory(
            timestamp=datetime.now(),
            emotional_valence=np.clip(emotional_valence, -1.0, 1.0),
            emotional_arousal=np.clip(emotional_arousal, 0.0, 1.0),
            visual_features=self.encode_visual(visual_input),
            auditory_features=self.encode_auditory(auditory_input),
            physiological_state=physiological_state or {},
            context=context or {}
        )
        
        # Consolidate the memory
        memory.strength = self.consolidator.compute_consolidation_strength(memory)
        
        # Store the memory
        self.memories.append(memory)
        
        # Prune old/weak memories if we exceed capacity
        if len(self.memories) > self.max_memories:
            self._prune_memories()
        
        return memory
    
    def retrieve_similar(
        self,
        query_visual: Optional[np.ndarray] = None,
        query_auditory: Optional[np.ndarray] = None,
        query_emotion: Optional[Tuple[float, float]] = None,
        top_k: int = 5
    ) -> List[Tuple[MultimodalMemory, float]]:
        """
        Retrieve memories similar to the query.
        
        Args:
            query_visual: Visual query
            query_auditory: Auditory query
            query_emotion: Emotional query (valence, arousal)
            top_k: Number of memories to retrieve
            
        Returns:
            List of (memory, similarity_score) tuples
        """
        if not self.memories:
            return []
        
        similarities = []
        
        for memory in self.memories:
            similarity = 0.0
            count = 0
            
            if query_visual is not None:
                visual_feat = self.encode_visual(query_visual)
                visual_sim = self._cosine_similarity(visual_feat, memory.visual_features)
                similarity += visual_sim
                count += 1
            
            if query_auditory is not None:
                auditory_feat = self.encode_auditory(query_auditory)
                auditory_sim = self._cosine_similarity(auditory_feat, memory.auditory_features)
                similarity += auditory_sim
                count += 1
            
            if query_emotion is not None:
                emotion_vec = np.array(query_emotion)
                memory_emotion = np.array([memory.emotional_valence, memory.emotional_arousal])
                emotion_sim = self._cosine_similarity(emotion_vec, memory_emotion)
                similarity += emotion_sim
                count += 1
            
            if count > 0:
                # Weight by memory strength
                avg_similarity = (similarity / count) * memory.strength
                similarities.append((memory, avg_similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def decay_memories(self):
        """Apply time-based decay to all memories."""
        for memory in self.memories:
            memory.strength *= (1.0 - self.decay_rate)
    
    def _prune_memories(self):
        """Remove weakest memories when capacity is exceeded."""
        # Sort by strength and keep the strongest
        self.memories.sort(key=lambda m: m.strength, reverse=True)
        self.memories = self.memories[:self.max_memories]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def get_memory_count(self) -> int:
        """Get the current number of stored memories."""
        return len(self.memories)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the memory system."""
        if not self.memories:
            return {
                'total_memories': 0,
                'avg_strength': 0.0,
                'avg_emotional_valence': 0.0,
                'avg_arousal': 0.0
            }
        
        strengths = [m.strength for m in self.memories]
        valences = [m.emotional_valence for m in self.memories]
        arousals = [m.emotional_arousal for m in self.memories]
        
        return {
            'total_memories': len(self.memories),
            'avg_strength': np.mean(strengths),
            'avg_emotional_valence': np.mean(valences),
            'avg_arousal': np.mean(arousals),
            'strongest_memory_strength': np.max(strengths),
            'weakest_memory_strength': np.min(strengths)
        }
