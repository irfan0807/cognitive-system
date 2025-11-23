"""
Embodied Cognition System

Main integration module where virtual biology drives virtual cognition.
Combines brain structures, neural networks, memory, and physiology into a unified system.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from cognitive_system.brain import VirtualBrain
from cognitive_system.neural import RealtimeNeuralNetwork
from cognitive_system.memory import MultimodalMemorySystem
from cognitive_system.physiology import VirtualPhysiology


@dataclass
class CognitiveState:
    """Represents the complete cognitive state of the system."""
    brain_state: Dict[str, float]
    physiological_state: Dict[str, any]
    cognitive_modulation: Dict[str, float]
    memory_stats: Dict[str, any]
    attention_focus: np.ndarray
    emotional_state: Tuple[float, float]  # (valence, arousal)


class EmbodiedCognitionSystem:
    """
    Complete embodied cognition system where virtual biology drives virtual cognition.
    
    This system integrates:
    - Virtual brain structures (brain stem, pituitary, PVN)
    - Neural networks for real-time learning
    - Multimodal memory (emotion, visual, auditory)
    - Virtual nervous system and physiology
    """
    
    def __init__(
        self,
        visual_dim: int = 64,
        auditory_dim: int = 32,
        attention_dim: int = 128,
        learning_rate: float = 0.01
    ):
        """
        Initialize the embodied cognition system.
        
        Args:
            visual_dim: Dimensionality of visual processing
            auditory_dim: Dimensionality of auditory processing
            attention_dim: Dimensionality of attention mechanism
            learning_rate: Learning rate for neural networks
        """
        # Core components
        self.brain = VirtualBrain()
        self.physiology = VirtualPhysiology()
        self.memory = MultimodalMemorySystem(visual_dim, auditory_dim)
        
        # Neural networks for real-time learning
        # Attention network: sensory -> attention focus
        sensory_input_dim = visual_dim + auditory_dim + 10  # +10 for emotional/physiological
        self.attention_network = RealtimeNeuralNetwork(
            [sensory_input_dim, attention_dim, attention_dim],
            learning_rate
        )
        
        # Decision network: attention + biology -> actions
        biology_dim = 20  # Brain and physiology state dimensions
        self.decision_network = RealtimeNeuralNetwork(
            [attention_dim + biology_dim, 64, 32],
            learning_rate
        )
        
        # Current state
        self.current_state: Optional[CognitiveState] = None
        self.timestep = 0
    
    def process_sensory_input(
        self,
        visual_input: np.ndarray,
        auditory_input: np.ndarray,
        social_context: float = 0.0,
        threat_level: float = 0.0,
        reward_signal: float = 0.0
    ) -> CognitiveState:
        """
        Process sensory input through the complete embodied cognition system.
        This is where virtual biology drives virtual cognition.
        
        Args:
            visual_input: Visual sensory input
            auditory_input: Auditory sensory input
            social_context: Social interaction intensity (0-1)
            threat_level: Perceived threat level (0-1)
            reward_signal: Reward/positive reinforcement (0-1)
            
        Returns:
            Complete cognitive state
        """
        # 1. BIOLOGY: Process through brain structures
        sensory_intensity = np.mean(np.abs(visual_input)) + np.mean(np.abs(auditory_input))
        sensory_intensity = np.clip(sensory_intensity, 0.0, 1.0)
        
        # Estimate emotional valence and arousal from inputs
        emotional_valence = reward_signal - threat_level
        emotional_arousal = sensory_intensity + abs(threat_level)
        
        brain_state = self.brain.process(
            sensory_input=sensory_intensity,
            social_stimulus=social_context,
            threat_level=threat_level,
            positive_emotion=max(0, emotional_valence)
        )
        
        # 2. PHYSIOLOGY: Update nervous system and physiology
        sensory_dict = {
            'visual': visual_input,
            'auditory': auditory_input
        }
        
        physiological_state = self.physiology.update(
            brain_state,
            sensory_dict,
            reward_signal
        )
        
        cognitive_modulation = physiological_state['cognitive_modulation']
        
        # 3. COGNITION DRIVEN BY BIOLOGY: Attention mechanism
        # Combine sensory and biological state for attention
        emotion_vec = np.array([emotional_valence, emotional_arousal])
        physio_vec = np.array([
            brain_state['arousal'],
            brain_state['heart_rate'] / 100.0,  # Normalize
            brain_state['oxytocin'],
            brain_state['stress_response'],
            cognitive_modulation['attention'],
            cognitive_modulation['mood'],
            cognitive_modulation['motivation'],
            cognitive_modulation['calmness']
        ])
        
        # Pad visual and auditory to expected dimensions
        visual_padded = self._pad_or_truncate(visual_input, self.memory.visual_dim)
        auditory_padded = self._pad_or_truncate(auditory_input, self.memory.auditory_dim)
        
        attention_input = np.concatenate([
            visual_padded,
            auditory_padded,
            emotion_vec,
            physio_vec
        ]).reshape(1, -1)
        
        # Attention is modulated by biological state
        attention_focus = self.attention_network.forward(attention_input)
        attention_focus = attention_focus.flatten()
        
        # Modulate attention by norepinephrine (attention neurotransmitter)
        attention_focus *= cognitive_modulation['attention']
        
        # 4. MEMORY: Form memories when emotionally salient
        memory_threshold = 0.3
        if abs(emotional_valence) > memory_threshold or emotional_arousal > memory_threshold:
            self.memory.form_memory(
                visual_padded,
                auditory_padded,
                emotional_valence,
                emotional_arousal,
                physiological_state={
                    **brain_state,
                    'energy': physiological_state['energy_level']
                },
                context={
                    'timestep': self.timestep,
                    'social_context': social_context,
                    'threat_level': threat_level
                }
            )
        
        # Decay memories over time
        if self.timestep % 10 == 0:  # Every 10 timesteps
            self.memory.decay_memories()
        
        # 5. Create cognitive state
        self.current_state = CognitiveState(
            brain_state=brain_state,
            physiological_state=physiological_state,
            cognitive_modulation=cognitive_modulation,
            memory_stats=self.memory.get_statistics(),
            attention_focus=attention_focus,
            emotional_state=(emotional_valence, emotional_arousal)
        )
        
        self.timestep += 1
        return self.current_state
    
    def make_decision(self, possible_actions: List[str]) -> Tuple[str, float]:
        """
        Make a decision based on current cognitive state.
        Biology drives cognition which drives behavior.
        
        Args:
            possible_actions: List of possible action names
            
        Returns:
            Tuple of (selected_action, confidence)
        """
        if self.current_state is None:
            # Default to first action if no state
            return possible_actions[0], 0.5
        
        # Combine attention and biological state for decision
        brain_vec = np.array([
            self.current_state.brain_state['arousal'],
            self.current_state.brain_state['heart_rate'] / 100.0,
            self.current_state.brain_state['oxytocin'],
            self.current_state.brain_state['stress_response'],
        ])
        
        physio_vec = np.array([
            self.current_state.physiological_state['energy_level'],
            self.current_state.physiological_state['homeostatic_balance'],
        ])
        
        modulation_vec = np.array([
            self.current_state.cognitive_modulation['attention'],
            self.current_state.cognitive_modulation['mood'],
            self.current_state.cognitive_modulation['motivation'],
            self.current_state.cognitive_modulation['calmness'],
        ])
        
        # Add emotional state
        emotion_vec = np.array(self.current_state.emotional_state)
        
        # Pad or truncate brain/physio to expected size
        biology_vec = np.concatenate([brain_vec, physio_vec, modulation_vec, emotion_vec])
        biology_padded = self._pad_or_truncate(biology_vec, 20)
        
        # Combine attention focus with biology
        decision_input = np.concatenate([
            self.current_state.attention_focus,
            biology_padded
        ]).reshape(1, -1)
        
        # Get decision from network
        decision_output = self.decision_network.forward(decision_input)
        decision_values = decision_output.flatten()
        
        # Map to actions
        num_actions = len(possible_actions)
        action_values = decision_values[:num_actions] if len(decision_values) >= num_actions else decision_values
        
        if len(action_values) < num_actions:
            # Pad with zeros if needed
            padded = np.zeros(num_actions)
            padded[:len(action_values)] = action_values
            action_values = padded
        
        # Select action with highest value
        best_action_idx = np.argmax(action_values)
        confidence = float(np.tanh(abs(action_values[best_action_idx])))
        
        return possible_actions[best_action_idx], confidence
    
    def learn_from_feedback(
        self,
        previous_input: Dict[str, np.ndarray],
        action_taken: str,
        outcome_reward: float
    ):
        """
        Learn from action outcomes using real-time learning.
        
        Args:
            previous_input: The sensory input that led to the action
            action_taken: The action that was taken
            outcome_reward: Reward received (positive for good, negative for bad)
        """
        # This would update the decision network based on outcomes
        # For now, we just process the reward through the system
        self.process_sensory_input(
            visual_input=previous_input.get('visual', np.zeros(10)),
            auditory_input=previous_input.get('auditory', np.zeros(10)),
            reward_signal=max(0, outcome_reward),
            threat_level=max(0, -outcome_reward)
        )
    
    def recall_similar_experiences(
        self,
        visual_query: Optional[np.ndarray] = None,
        auditory_query: Optional[np.ndarray] = None,
        emotional_query: Optional[Tuple[float, float]] = None,
        top_k: int = 3
    ) -> List[Tuple[any, float]]:
        """
        Recall similar experiences from memory.
        
        Args:
            visual_query: Visual query
            auditory_query: Auditory query
            emotional_query: Emotional query (valence, arousal)
            top_k: Number of memories to retrieve
            
        Returns:
            List of similar memories with similarity scores
        """
        return self.memory.retrieve_similar(
            visual_query, auditory_query, emotional_query, top_k
        )
    
    def get_state_summary(self) -> Dict[str, any]:
        """
        Get a summary of the current system state.
        
        Returns:
            Dictionary with state summary
        """
        if self.current_state is None:
            return {'status': 'not_initialized'}
        
        return {
            'timestep': self.timestep,
            'arousal': self.current_state.brain_state['arousal'],
            'heart_rate': self.current_state.brain_state['heart_rate'],
            'oxytocin': self.current_state.brain_state['oxytocin'],
            'stress': self.current_state.brain_state['stress_response'],
            'energy': self.current_state.physiological_state['energy_level'],
            'homeostasis': self.current_state.physiological_state['homeostatic_balance'],
            'attention_level': self.current_state.cognitive_modulation['attention'],
            'mood': self.current_state.cognitive_modulation['mood'],
            'motivation': self.current_state.cognitive_modulation['motivation'],
            'emotional_valence': self.current_state.emotional_state[0],
            'emotional_arousal': self.current_state.emotional_state[1],
            'total_memories': self.current_state.memory_stats['total_memories']
        }
    
    @staticmethod
    def _pad_or_truncate(arr: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate array to target length."""
        if len(arr) >= target_length:
            return arr[:target_length]
        else:
            padded = np.zeros(target_length)
            padded[:len(arr)] = arr
            return padded
