"""
Behavior Generation Engine

The complex virtual brain model generates all the character's behaviors in
real time. This includes motor behaviors, expressive behaviors, and
environmental interactions.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging


class BehaviorEngine:
    """
    Behavior generation engine driven by the virtual brain.
    
    Generates all character behaviors in real time based on:
    - Neural network outputs
    - Physiological state
    - Environmental context
    - Learned associations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Behavior Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Behavior state
        self.current_behaviors = {}
        self.behavior_history = []
        
        # Motor control
        self.motor_state = np.zeros(10)
        self.facial_expression = np.zeros(6)  # 6 basic emotions
        
        # Environmental interaction state
        self.interaction_context = None
        self.attention_focus = None
        
        self.logger.info("Behavior Engine initialized")
    
    def generate_behaviors(
        self,
        neural_response: Dict[str, Any],
        physiological_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate all behaviors in real time.
        
        Args:
            neural_response: Neural network outputs
            physiological_state: Current physiological state
            
        Returns:
            Dictionary of generated behaviors
        """
        # Generate motor behaviors
        motor_behaviors = self._generate_motor_behaviors(
            neural_response, physiological_state
        )
        
        # Generate expressive behaviors (facial, vocal)
        expressive_behaviors = self._generate_expressive_behaviors(
            physiological_state
        )
        
        # Generate attention/orientation behaviors
        attention_behaviors = self._generate_attention_behaviors(
            neural_response
        )
        
        # Combine all behaviors
        behaviors = {
            'motor': motor_behaviors,
            'expressive': expressive_behaviors,
            'attention': attention_behaviors,
            'timestamp': len(self.behavior_history),
        }
        
        # Store in history
        self.current_behaviors = behaviors
        self.behavior_history.append(behaviors)
        
        return behaviors
    
    def _generate_motor_behaviors(
        self,
        neural_response: Dict[str, Any],
        physiological_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate motor behaviors (movement, gestures, etc.).
        
        Args:
            neural_response: Neural outputs
            physiological_state: Physiological state
            
        Returns:
            Motor behavior commands
        """
        # Extract motor commands from neural response
        motor_commands = neural_response.get('motor_commands', np.zeros(10))
        
        if isinstance(motor_commands, (list, tuple)):
            motor_commands = np.array(motor_commands)
        elif not isinstance(motor_commands, np.ndarray):
            motor_commands = np.zeros(10)
        
        # Modulate by arousal and stress
        arousal = physiological_state.get('arousal', 0.5)
        stress = physiological_state.get('stress_level', 0.0)
        
        # Higher arousal increases motor intensity
        intensity_modulation = 0.5 + arousal * 0.5
        motor_commands = motor_commands * intensity_modulation
        
        # Update motor state with smoothing
        self.motor_state = self.motor_state * 0.7 + motor_commands[:10] * 0.3
        
        return {
            'position_commands': self.motor_state.copy(),
            'intensity': intensity_modulation,
            'tremor': stress * 0.2,  # Stress causes tremor
        }
    
    def _generate_expressive_behaviors(
        self,
        physiological_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate expressive behaviors (facial expressions, vocalizations).
        
        Args:
            physiological_state: Physiological state
            
        Returns:
            Expressive behavior commands
        """
        # Extract emotional state
        emotional_state = physiological_state.get('emotional_state', {})
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.5)
        stress = physiological_state.get('stress_level', 0.0)
        
        # Map to facial expressions (6 basic emotions)
        # [happy, sad, angry, fearful, surprised, disgusted]
        target_expression = np.zeros(6)
        
        if valence > 0.3:
            target_expression[0] = valence * arousal  # Happy
        elif valence < -0.3:
            if stress > 0.5:
                target_expression[3] = abs(valence) * arousal  # Fearful
            else:
                target_expression[1] = abs(valence) * arousal  # Sad
        
        if stress > 0.7:
            target_expression[2] = stress * arousal  # Angry
        
        # Smooth transition to target expression
        self.facial_expression = self.facial_expression * 0.8 + target_expression * 0.2
        
        # Normalize
        expr_sum = np.sum(self.facial_expression)
        if expr_sum > 0:
            self.facial_expression = self.facial_expression / expr_sum
        
        return {
            'facial_expression': self.facial_expression.copy(),
            'vocal_tone': {
                'pitch': 1.0 + valence * 0.3,
                'volume': 0.5 + arousal * 0.5,
                'rate': 1.0 + stress * 0.4,
            }
        }
    
    def _generate_attention_behaviors(
        self,
        neural_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate attention and orientation behaviors.
        
        Args:
            neural_response: Neural outputs
            
        Returns:
            Attention behavior commands
        """
        # Extract attention from neural response
        attention = neural_response.get('attention', np.zeros(5))
        
        if isinstance(attention, (list, tuple)):
            attention = np.array(attention)
        elif not isinstance(attention, np.ndarray):
            attention = np.zeros(5)
        
        # Determine focus point
        if len(attention) > 0:
            focus_idx = np.argmax(attention)
            focus_strength = attention[focus_idx] if len(attention) > focus_idx else 0.0
        else:
            focus_idx = 0
            focus_strength = 0.0
        
        self.attention_focus = {
            'target': focus_idx,
            'strength': float(focus_strength),
        }
        
        return {
            'gaze_target': focus_idx,
            'attention_strength': float(focus_strength),
            'head_orientation': attention[:3].tolist() if len(attention) >= 3 else [0.0, 0.0, 0.0],
        }
    
    def interact_with_environment(
        self,
        environment_type: str,
        environment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Allow the character to interact with its environment.
        
        This supports interactions such as:
        - Watching videos
        - Playing games on a simulated screen
        - Manipulating objects
        
        Args:
            environment_type: Type of environment ('video', 'game', 'object', etc.)
            environment_data: Data about the environment
            
        Returns:
            Interaction results
        """
        self.interaction_context = {
            'type': environment_type,
            'data': environment_data,
            'timestamp': len(self.behavior_history),
        }
        
        interaction_result = {}
        
        if environment_type == 'video':
            # Watch video - track visual attention
            interaction_result = self._interact_with_video(environment_data)
        
        elif environment_type == 'game':
            # Play game - generate interactive responses
            interaction_result = self._interact_with_game(environment_data)
        
        elif environment_type == 'music':
            # Respond to music - generate rhythmic movements
            interaction_result = self._interact_with_music(environment_data)
        
        elif environment_type == 'object':
            # Manipulate object
            interaction_result = self._interact_with_object(environment_data)
        
        self.logger.info(f"Interacting with {environment_type}")
        
        return interaction_result
    
    def _interact_with_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Interact with video content."""
        # Track salient features
        visual_saliency = video_data.get('saliency', np.zeros(10))
        
        return {
            'attention_allocation': visual_saliency,
            'engagement': np.mean(visual_saliency) if len(visual_saliency) > 0 else 0.0,
        }
    
    def _interact_with_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Interact with game."""
        # Generate game actions based on current state
        game_state = game_data.get('state', {})
        
        # Simple action selection
        action_space = game_data.get('action_space', [])
        if action_space:
            # Select action based on motor state
            action_idx = int(np.argmax(np.abs(self.motor_state))) % len(action_space)
            selected_action = action_space[action_idx]
        else:
            selected_action = None
        
        return {
            'action': selected_action,
            'engagement': 1.0 if selected_action else 0.0,
        }
    
    def _interact_with_music(self, music_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Respond to music by automatically responding to the beat.
        
        The character's stimulus system is affected by music, generating
        corresponding movement patterns.
        """
        # Extract beat information
        beat_strength = music_data.get('beat_strength', 0.0)
        beat_frequency = music_data.get('beat_frequency', 1.0)
        
        # Generate rhythmic motor response
        time_phase = len(self.behavior_history) * beat_frequency
        rhythmic_response = np.sin(time_phase) * beat_strength
        
        # Modulate motor state
        rhythm_motor = self.motor_state.copy()
        for i in range(len(rhythm_motor)):
            phase_offset = (i / len(rhythm_motor)) * 2 * np.pi
            rhythm_motor[i] = rhythmic_response * np.sin(time_phase + phase_offset)
        
        return {
            'rhythmic_movement': rhythm_motor.tolist(),
            'beat_synchronization': beat_strength,
            'movement_frequency': beat_frequency,
        }
    
    def _interact_with_object(self, object_data: Dict[str, Any]) -> Dict[str, Any]:
        """Interact with objects."""
        # Generate manipulation actions
        object_position = object_data.get('position', [0, 0, 0])
        
        # Reach toward object
        reach_vector = np.array(object_position[:3]) - self.motor_state[:3]
        
        return {
            'reach_vector': reach_vector.tolist(),
            'grasp_strength': 0.5,
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current behavior state."""
        return {
            'current_behaviors': self.current_behaviors,
            'behavior_count': len(self.behavior_history),
            'motor_state': self.motor_state.tolist(),
            'facial_expression': self.facial_expression.tolist(),
            'attention_focus': self.attention_focus,
        }
