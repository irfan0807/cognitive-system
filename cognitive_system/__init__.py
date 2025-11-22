"""
Cognitive System - Autonomous Embodied AI Simulation Framework

A comprehensive framework for creating autonomous characters based on embodied cognition,
modeled after the Baby X system. Features real-time neural network simulation,
virtual biology, and experiential learning.
"""

__version__ = "0.1.0"
__author__ = "Cognitive System Team"

from .core.embodied_cognition import EmbodiedCognitionSystem
from .virtual_biology.nervous_system import VirtualNervousSystem
from .virtual_biology.brain import VirtualBrain
from .memory.multimodal_memory import MultimodalMemorySystem
from .behavior.behavior_engine import BehaviorEngine

__all__ = [
    "EmbodiedCognitionSystem",
    "VirtualNervousSystem",
    "VirtualBrain",
    "MultimodalMemorySystem",
    "BehaviorEngine",
]
