"""
Multimodal Memory System

Implements the memory construction system that builds a history of memories
from events encountered through life. Memories are reconstructed multimodally,
integrating emotion, visual stimuli, and auditory stimulation at the time
of memory formation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging


@dataclass
class MultimodalMemory:
    """
    A multimodal memory integrating multiple sensory and emotional modalities.
    
    When a memory is formed (e.g., emotional conditioning by a word like "fire"),
    it is reconstructed with:
    - Emotion at the time
    - Visual stimuli at the time
    - Auditory stimulation at the time
    """
    
    timestamp: float
    
    # Emotional context
    emotional_valence: float  # Positive/negative emotion
    emotional_arousal: float  # Intensity of emotion
    stress_level: float
    
    # Visual modality
    visual_data: np.ndarray = field(default_factory=lambda: np.zeros(10))
    visual_attention: Optional[np.ndarray] = None
    
    # Auditory modality
    auditory_data: np.ndarray = field(default_factory=lambda: np.zeros(10))
    auditory_intensity: float = 0.0
    
    # Semantic/conceptual
    semantic_tags: List[str] = field(default_factory=list)
    
    # Physiological state
    heart_rate: float = 60.0
    hormone_levels: Dict[str, float] = field(default_factory=dict)
    
    # Behavioral context
    behaviors: Dict[str, Any] = field(default_factory=dict)
    
    # Memory strength (consolidation)
    strength: float = 1.0
    
    def decay(self, decay_rate: float):
        """Apply decay to memory strength over time."""
        self.strength *= (1.0 - decay_rate)
    
    def consolidate(self, consolidation_boost: float):
        """Strengthen memory through consolidation."""
        self.strength = min(1.0, self.strength + consolidation_boost)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            'timestamp': self.timestamp,
            'emotional_valence': self.emotional_valence,
            'emotional_arousal': self.emotional_arousal,
            'stress_level': self.stress_level,
            'visual_data': self.visual_data.tolist() if isinstance(self.visual_data, np.ndarray) else self.visual_data,
            'auditory_data': self.auditory_data.tolist() if isinstance(self.auditory_data, np.ndarray) else self.auditory_data,
            'semantic_tags': self.semantic_tags,
            'strength': self.strength,
        }


class MultimodalMemorySystem:
    """
    Memory system that constructs a history of memories multimodally.
    
    Integrates emotional, visual, and auditory information when forming
    memories, enabling rich experiential learning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Multimodal Memory System.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Memory storage
        self.episodic_memories: List[MultimodalMemory] = []
        self.semantic_memory: Dict[str, Any] = {}
        
        # Memory parameters
        self.decay_rate = self.config.get('decay_rate', 0.01)
        self.consolidation_threshold = self.config.get('consolidation_threshold', 0.7)
        
        self.logger.info("Multimodal Memory System initialized")
    
    def store_experience(self, experience: Dict[str, Any]) -> MultimodalMemory:
        """
        Store an experience as a multimodal memory.
        
        Reconstructs the experience multimodally by integrating:
        - Emotion at the time
        - Visual stimuli at the time
        - Auditory stimulation at the time
        
        Args:
            experience: Experience dictionary from embodied cognition system
            
        Returns:
            Created memory object
        """
        # Extract multimodal components
        emotional_state = experience.get('emotional_state', {})
        sensory_input = experience.get('sensory_input', {})
        physiological_state = experience.get('physiological_state', {})
        
        # Create multimodal memory
        memory = MultimodalMemory(
            timestamp=experience.get('timestamp', 0.0),
            
            # Emotional modality
            emotional_valence=emotional_state.get('valence', 0.0),
            emotional_arousal=emotional_state.get('arousal', 0.5),
            stress_level=physiological_state.get('stress_level', 0.0),
            
            # Visual modality
            visual_data=self._encode_visual(sensory_input.get('visual', None)),
            visual_attention=sensory_input.get('visual', {}).get('attention_point', None) if isinstance(sensory_input.get('visual'), dict) else None,
            
            # Auditory modality
            auditory_data=self._encode_auditory(sensory_input.get('auditory', None)),
            auditory_intensity=self._get_auditory_intensity(sensory_input.get('auditory', None)),
            
            # Physiological
            heart_rate=physiological_state.get('heart_rate', 60.0),
            hormone_levels=physiological_state.get('chemical_levels', {}).copy(),
            
            # Behavioral
            behaviors=experience.get('behaviors', {}).copy(),
            
            # Initial strength based on emotional intensity
            strength=min(1.0, abs(emotional_state.get('valence', 0.0)) + emotional_state.get('arousal', 0.5))
        )
        
        # Store memory
        self.episodic_memories.append(memory)
        
        # Detect semantic patterns (e.g., emotional conditioning)
        self._extract_semantic_associations(memory)
        
        self.logger.debug(f"Stored multimodal memory at t={memory.timestamp}")
        
        return memory
    
    def _encode_visual(self, visual_input: Any) -> np.ndarray:
        """Encode visual input for memory storage."""
        if visual_input is None:
            return np.zeros(10)
        elif isinstance(visual_input, np.ndarray):
            return visual_input.flatten()[:10]
        elif isinstance(visual_input, (list, tuple)):
            return np.array(visual_input).flatten()[:10]
        elif isinstance(visual_input, dict):
            # Extract relevant visual features
            features = visual_input.get('features', np.zeros(10))
            if isinstance(features, (list, tuple)):
                return np.array(features)[:10]
            elif isinstance(features, np.ndarray):
                return features.flatten()[:10]
        return np.zeros(10)
    
    def _encode_auditory(self, auditory_input: Any) -> np.ndarray:
        """Encode auditory input for memory storage."""
        if auditory_input is None:
            return np.zeros(10)
        elif isinstance(auditory_input, np.ndarray):
            return auditory_input.flatten()[:10]
        elif isinstance(auditory_input, (list, tuple)):
            return np.array(auditory_input).flatten()[:10]
        elif isinstance(auditory_input, dict):
            features = auditory_input.get('features', np.zeros(10))
            if isinstance(features, (list, tuple)):
                return np.array(features)[:10]
            elif isinstance(features, np.ndarray):
                return features.flatten()[:10]
        return np.zeros(10)
    
    def _get_auditory_intensity(self, auditory_input: Any) -> float:
        """Get intensity of auditory input."""
        if auditory_input is None:
            return 0.0
        elif isinstance(auditory_input, (int, float)):
            return float(np.clip(auditory_input, 0.0, 1.0))
        elif isinstance(auditory_input, np.ndarray):
            return float(np.clip(np.mean(np.abs(auditory_input)), 0.0, 1.0))
        elif isinstance(auditory_input, dict):
            return auditory_input.get('intensity', 0.0)
        return 0.0
    
    def _extract_semantic_associations(self, memory: MultimodalMemory):
        """
        Extract semantic associations from episodic memory.
        
        For example, if "fire" is associated with strong negative emotion,
        this creates a semantic association.
        """
        # Check for strong emotional memories
        if abs(memory.emotional_valence) > self.consolidation_threshold:
            # Create semantic association
            emotion_type = "positive" if memory.emotional_valence > 0 else "negative"
            
            # Store in semantic memory
            for tag in memory.semantic_tags:
                if tag not in self.semantic_memory:
                    self.semantic_memory[tag] = {
                        'positive_associations': 0,
                        'negative_associations': 0,
                        'total_encounters': 0,
                    }
                
                self.semantic_memory[tag]['total_encounters'] += 1
                if emotion_type == "positive":
                    self.semantic_memory[tag]['positive_associations'] += 1
                else:
                    self.semantic_memory[tag]['negative_associations'] += 1
    
    def recall_memories(
        self, 
        cue: Dict[str, Any],
        top_k: int = 5
    ) -> List[MultimodalMemory]:
        """
        Recall memories based on a cue.
        
        Uses multimodal similarity to find relevant memories.
        
        Args:
            cue: Cue dictionary (can contain visual, auditory, emotional, etc.)
            top_k: Number of memories to recall
            
        Returns:
            List of most relevant memories
        """
        if not self.episodic_memories:
            return []
        
        # Calculate similarity for each memory
        similarities = []
        for memory in self.episodic_memories:
            similarity = self._calculate_similarity(memory, cue)
            # Weight by memory strength
            weighted_similarity = similarity * memory.strength
            similarities.append((weighted_similarity, memory))
        
        # Sort by similarity and return top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        recalled = [mem for _, mem in similarities[:top_k]]
        
        # Consolidate recalled memories (reconsolidation)
        for memory in recalled:
            memory.consolidate(0.05)
        
        return recalled
    
    def _calculate_similarity(
        self, 
        memory: MultimodalMemory,
        cue: Dict[str, Any]
    ) -> float:
        """
        Calculate multimodal similarity between memory and cue.
        
        Integrates similarity across visual, auditory, and emotional modalities.
        """
        similarity = 0.0
        num_modalities = 0
        
        # Visual similarity
        if 'visual' in cue:
            visual_cue = self._encode_visual(cue['visual'])
            visual_sim = self._cosine_similarity(memory.visual_data, visual_cue)
            similarity += visual_sim
            num_modalities += 1
        
        # Auditory similarity
        if 'auditory' in cue:
            auditory_cue = self._encode_auditory(cue['auditory'])
            auditory_sim = self._cosine_similarity(memory.auditory_data, auditory_cue)
            similarity += auditory_sim
            num_modalities += 1
        
        # Emotional similarity
        if 'emotional_state' in cue:
            emotional_state = cue['emotional_state']
            valence_diff = abs(memory.emotional_valence - emotional_state.get('valence', 0.0))
            arousal_diff = abs(memory.emotional_arousal - emotional_state.get('arousal', 0.5))
            emotional_sim = 1.0 - (valence_diff + arousal_diff) / 2.0
            similarity += emotional_sim
            num_modalities += 1
        
        # Average similarity across modalities
        if num_modalities > 0:
            similarity /= num_modalities
        
        return similarity
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            # Pad shorter vector
            max_len = max(len(vec1), len(vec2))
            vec1 = np.pad(vec1, (0, max_len - len(vec1)))
            vec2 = np.pad(vec2, (0, max_len - len(vec2)))
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def update(self, delta_time: float):
        """
        Update memory system (apply decay, consolidation, etc.).
        
        Args:
            delta_time: Time step
        """
        # Apply decay to all memories
        for memory in self.episodic_memories:
            memory.decay(self.decay_rate * delta_time)
        
        # Remove very weak memories
        self.episodic_memories = [
            m for m in self.episodic_memories if m.strength > 0.1
        ]
    
    def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        return len(self.episodic_memories)
    
    def get_semantic_associations(self, concept: str) -> Optional[Dict[str, Any]]:
        """Get semantic associations for a concept."""
        return self.semantic_memory.get(concept, None)
