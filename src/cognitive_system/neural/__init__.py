"""
Neural Network Components for Real-time Learning

This module provides neural network components for real-time learning
in the embodied cognition system.
"""

import numpy as np
from typing import Optional, Tuple


class NeuralLayer:
    """A single neural network layer with online learning capability."""
    
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.01):
        """
        Initialize a neural layer.
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            learning_rate: Learning rate for weight updates
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases with small random values
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
        
        # Cache for backpropagation
        self.last_input: Optional[np.ndarray] = None
        self.last_output: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Output array of shape (batch_size, output_size)
        """
        self.last_input = x
        self.last_output = np.dot(x, self.weights) + self.biases
        return self.last_output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass for gradient computation and weight update.
        
        Args:
            gradient: Gradient from the next layer
            
        Returns:
            Gradient to pass to the previous layer
        """
        if self.last_input is None:
            raise ValueError("Must call forward before backward")
        
        # Compute gradients
        input_gradient = np.dot(gradient, self.weights.T)
        weight_gradient = np.dot(self.last_input.T, gradient)
        bias_gradient = np.sum(gradient, axis=0)
        
        # Update weights and biases (online learning)
        self.weights -= self.learning_rate * weight_gradient
        self.biases -= self.learning_rate * bias_gradient
        
        return input_gradient


class ActivationFunction:
    """Activation functions for neural networks."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)


class RealtimeNeuralNetwork:
    """
    Neural network with real-time learning capabilities.
    Supports online learning for embodied cognition.
    """
    
    def __init__(self, layer_sizes: list, learning_rate: float = 0.01):
        """
        Initialize a multi-layer neural network.
        
        Args:
            layer_sizes: List of layer sizes (e.g., [10, 20, 10] for 3 layers)
            learning_rate: Learning rate for all layers
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                NeuralLayer(layer_sizes[i], layer_sizes[i + 1], learning_rate)
            )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            x: Input array
            
        Returns:
            Network output
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
            output = ActivationFunction.tanh(output)  # Apply activation
        return output
    
    def train_online(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Perform online learning with a single sample.
        
        Args:
            x: Input sample
            target: Target output
            
        Returns:
            Loss value
        """
        # Forward pass
        output = self.forward(x)
        
        # Compute loss (MSE)
        loss = np.mean((output - target) ** 2)
        
        # Backward pass
        gradient = 2 * (output - target) / output.size
        
        # Backpropagate through layers (in reverse)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        
        return loss
