"""
Neural Network Controller

Implements the simplified brain simulation model that animates the character live.
Uses neural networks for real-time processing and learning.
"""

import numpy as np
from typing import Dict, Any, List
import logging


class NeuralNetworkController:
    """
    Neural network controller for the embodied cognition system.
    
    Provides real-time neural processing that drives character animation
    and behavior generation.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the neural network controller.
        
        Args:
            input_size: Size of input layer (sensory + physiological)
            hidden_size: Size of hidden layer
            output_size: Size of output layer (motor commands)
        """
        self.logger = logging.getLogger(__name__)
        
        # Simple feedforward network with adaptation capability
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)
        
        # Store for learning
        self.last_activation_hidden = None
        self.last_activation_output = None
        
        self.logger.info(f"Neural network initialized: {input_size} -> {hidden_size} -> {output_size}")
    
    def forward(
        self, 
        sensory_input: Dict[str, Any],
        physiological_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward pass through the neural network.
        
        Processes sensory input and physiological state to generate
        neural responses that drive behavior.
        
        Args:
            sensory_input: Sensory inputs from environment
            physiological_state: Current physiological state
            
        Returns:
            Neural response dictionary
        """
        # Convert inputs to vector
        input_vector = self._encode_inputs(sensory_input, physiological_state)
        
        # Forward pass
        hidden = self._activate(
            np.dot(input_vector, self.weights_input_hidden) + self.bias_hidden
        )
        output = self._activate(
            np.dot(hidden, self.weights_hidden_output) + self.bias_output
        )
        
        # Store for learning
        self.last_activation_hidden = hidden
        self.last_activation_output = output
        
        return {
            'motor_commands': output[:10] if len(output) > 10 else output,
            'attention': output[10:15] if len(output) > 15 else np.zeros(5),
            'arousal': float(np.mean(hidden)),
        }
    
    def _encode_inputs(
        self, 
        sensory_input: Dict[str, Any],
        physiological_state: Dict[str, Any]
    ) -> np.ndarray:
        """
        Encode sensory and physiological inputs into a vector.
        
        Args:
            sensory_input: Sensory inputs
            physiological_state: Physiological state
            
        Returns:
            Input vector for neural network
        """
        # Simple encoding - in practice would be more sophisticated
        visual = sensory_input.get('visual', np.zeros(10))
        auditory = sensory_input.get('auditory', np.zeros(10))
        
        # Ensure numpy arrays
        if not isinstance(visual, np.ndarray):
            visual = np.array(visual) if hasattr(visual, '__iter__') else np.zeros(10)
        if not isinstance(auditory, np.ndarray):
            auditory = np.array(auditory) if hasattr(auditory, '__iter__') else np.zeros(10)
        
        # Physiological encoding
        heart_rate = physiological_state.get('heart_rate', 60.0) / 100.0
        stress = physiological_state.get('stress_level', 0.0)
        
        input_vector = np.concatenate([
            visual.flatten()[:10],
            auditory.flatten()[:10],
            [heart_rate, stress]
        ])
        
        return input_vector
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Activation function (tanh)."""
        return np.tanh(x)
    
    def adapt(self, experience: Dict[str, Any], learning_rate: float):
        """
        Adapt network weights based on experience.
        
        This enables real-time learning from experiences.
        
        Args:
            experience: Experience dictionary
            learning_rate: Learning rate for adaptation
        """
        # Simple hebbian-style learning
        # In practice, would use more sophisticated learning rules
        if self.last_activation_hidden is not None:
            # Strengthen connections based on experience
            reward = experience.get('emotional_state', {}).get('valence', 0.0)
            
            if abs(reward) > 0.1:  # Only learn from significant experiences
                # Update weights (simplified)
                delta_hidden_output = learning_rate * reward * np.outer(
                    self.last_activation_hidden,
                    self.last_activation_output
                )
                self.weights_hidden_output += delta_hidden_output
                
                self.logger.debug(f"Network adapted with reward {reward}")
