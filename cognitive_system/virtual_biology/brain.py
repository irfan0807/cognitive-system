"""
Virtual Brain Model

Implements the virtual brain modeled live with specific functional structures
and areas including the brainstem, nuclei, and glands responsible for chemical
regulation.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging


class BrainstemStructure:
    """
    Brain stem structure connecting brain to body.
    
    Governs how the character acts, moves, and responds. Contains critical
    nuclei for motor control and autonomic functions.
    """
    
    def __init__(self):
        """Initialize brainstem structure."""
        self.logger = logging.getLogger(__name__)
        
        # Oculomotor nuclei - controls eye movements
        self.oculomotor_nuclei = OculomotorNuclei()
        
        # Autonomic control centers
        self.respiratory_center = {'rate': 12.0, 'depth': 1.0}
        self.cardiovascular_center = {'heart_rate': 60.0, 'blood_pressure': 120.0}
        
        # Motor pathways
        self.motor_pathways = np.zeros(10)
        
        self.logger.info("Brainstem initialized")
    
    def process(
        self, 
        higher_brain_input: Dict[str, Any],
        sensory_input: Dict[str, Any],
        delta_time: float
    ) -> Dict[str, Any]:
        """
        Process inputs and generate motor outputs.
        
        Args:
            higher_brain_input: Input from higher brain areas
            sensory_input: Direct sensory input
            delta_time: Time step
            
        Returns:
            Motor and autonomic outputs
        """
        # Update oculomotor control
        eye_movements = self.oculomotor_nuclei.generate_eye_movements(
            sensory_input.get('visual', {}), delta_time
        )
        
        # Update autonomic functions
        self._update_autonomic_functions(higher_brain_input, delta_time)
        
        # Generate motor commands
        motor_output = self._generate_motor_output(higher_brain_input)
        
        return {
            'eye_movements': eye_movements,
            'respiratory_state': self.respiratory_center.copy(),
            'cardiovascular_state': self.cardiovascular_center.copy(),
            'motor_commands': motor_output,
        }
    
    def _update_autonomic_functions(
        self, 
        higher_input: Dict[str, Any],
        delta_time: float
    ):
        """Update autonomic nervous system functions."""
        # Adjust based on emotional/stress state
        stress = higher_input.get('stress_level', 0.0)
        arousal = higher_input.get('arousal', 0.5)
        
        # Update respiratory center
        self.respiratory_center['rate'] = 12.0 + stress * 18.0
        self.respiratory_center['depth'] = 1.0 + arousal * 0.5
        
        # Update cardiovascular center
        self.cardiovascular_center['heart_rate'] = 60.0 + stress * 60.0
        self.cardiovascular_center['blood_pressure'] = 120.0 + arousal * 20.0
    
    def _generate_motor_output(self, higher_input: Dict[str, Any]) -> np.ndarray:
        """Generate motor pathway outputs."""
        motor_commands = higher_input.get('motor_intent', np.zeros(10))
        
        # Brainstem modulates and relays motor commands
        if isinstance(motor_commands, (list, tuple)):
            motor_commands = np.array(motor_commands)
        elif not isinstance(motor_commands, np.ndarray):
            motor_commands = np.zeros(10)
        
        # Apply brainstem processing
        self.motor_pathways = motor_commands * 0.9 + self.motor_pathways * 0.1
        
        return self.motor_pathways


class OculomotorNuclei:
    """
    Oculomotor nuclei in the brainstem.
    
    Controls eye movements and visual tracking.
    """
    
    def __init__(self):
        """Initialize oculomotor nuclei."""
        self.eye_position = np.array([0.0, 0.0])  # [horizontal, vertical]
        self.target_position = np.array([0.0, 0.0])
    
    def generate_eye_movements(
        self, 
        visual_input: Any,
        delta_time: float
    ) -> Dict[str, Any]:
        """
        Generate eye movements based on visual input.
        
        Args:
            visual_input: Visual stimuli
            delta_time: Time step
            
        Returns:
            Eye movement commands
        """
        # Simple saccade model - eyes move toward salient stimuli
        if isinstance(visual_input, dict):
            self.target_position = np.array(
                visual_input.get('attention_point', [0.0, 0.0])
            )[:2]
        
        # Smooth pursuit of target
        error = self.target_position - self.eye_position
        movement = error * 0.5 * delta_time
        self.eye_position += movement
        
        # Clamp to reasonable range [-1, 1]
        self.eye_position = np.clip(self.eye_position, -1.0, 1.0)
        
        return {
            'position': self.eye_position.copy(),
            'velocity': movement,
        }


class PituitaryGland:
    """
    Pituitary gland responsible for chemical regulation.
    
    Releases simulated substances like oxytocin and other hormones
    that regulate physiological and emotional states.
    """
    
    def __init__(self):
        """Initialize pituitary gland."""
        self.logger = logging.getLogger(__name__)
        
        # Hormone levels
        self.oxytocin = 0.5
        self.vasopressin = 0.5
        self.acth = 0.3  # Adrenocorticotropic hormone
        
        # Release rates
        self.release_rates = {
            'oxytocin': 0.0,
            'vasopressin': 0.0,
            'acth': 0.0,
        }
        
        self.logger.info("Pituitary gland initialized")
    
    def release_hormones(
        self, 
        hypothalamic_input: Dict[str, Any],
        delta_time: float
    ) -> Dict[str, float]:
        """
        Release hormones based on hypothalamic signals.
        
        Args:
            hypothalamic_input: Signals from hypothalamus
            delta_time: Time step
            
        Returns:
            Current hormone levels
        """
        # Oxytocin release (social bonding, stress reduction)
        social_signal = hypothalamic_input.get('social_stimulation', 0.0)
        self.release_rates['oxytocin'] = social_signal * 0.1
        self.oxytocin += self.release_rates['oxytocin'] * delta_time
        
        # ACTH release (stress response)
        stress_signal = hypothalamic_input.get('stress_level', 0.0)
        self.release_rates['acth'] = stress_signal * 0.1
        self.acth += self.release_rates['acth'] * delta_time
        
        # Natural decay
        self.oxytocin *= (1.0 - 0.05 * delta_time)
        self.acth *= (1.0 - 0.05 * delta_time)
        
        # Clamp values
        self.oxytocin = np.clip(self.oxytocin, 0.0, 1.0)
        self.acth = np.clip(self.acth, 0.0, 1.0)
        
        return {
            'oxytocin': self.oxytocin,
            'vasopressin': self.vasopressin,
            'acth': self.acth,
        }


class ParaventricularNucleus:
    """
    Paraventricular nucleus (PVN) in the hypothalamus.
    
    Key area for stress response and autonomic regulation.
    Communicates with pituitary gland for hormone release.
    """
    
    def __init__(self):
        """Initialize paraventricular nucleus."""
        self.logger = logging.getLogger(__name__)
        
        # Neural activity level
        self.activity = 0.5
        
        # Outputs to pituitary
        self.pituitary_signals = {
            'oxytocin_release': 0.0,
            'acth_release': 0.0,
        }
        
        self.logger.info("Paraventricular nucleus initialized")
    
    def process(
        self, 
        emotional_state: Dict[str, Any],
        physiological_state: Dict[str, Any],
        delta_time: float
    ) -> Dict[str, float]:
        """
        Process emotional and physiological inputs.
        
        Args:
            emotional_state: Current emotional state
            physiological_state: Current physiological state
            delta_time: Time step
            
        Returns:
            Signals to pituitary gland
        """
        # Activity increases with stress and arousal
        stress = physiological_state.get('stress_level', 0.0)
        arousal = physiological_state.get('arousal', 0.5)
        valence = emotional_state.get('valence', 0.0)
        
        target_activity = (stress + arousal) / 2.0
        self.activity += (target_activity - self.activity) * 0.1 * delta_time
        
        # Generate pituitary signals
        # Positive valence increases oxytocin signal
        self.pituitary_signals['oxytocin_release'] = max(0.0, valence) * self.activity
        
        # Stress increases ACTH signal
        self.pituitary_signals['acth_release'] = stress * self.activity
        
        return self.pituitary_signals.copy()


class VirtualBrain:
    """
    Virtual Brain integrating all brain structures.
    
    Models the brain live with specific functional structures:
    - Brainstem (motor control, autonomic functions)
    - Oculomotor nuclei (eye movements)
    - Pituitary gland (hormone release, oxytocin)
    - Paraventricular nucleus (stress response)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the virtual brain.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Brain structures
        self.brainstem = BrainstemStructure()
        self.pituitary_gland = PituitaryGland()
        self.paraventricular_nucleus = ParaventricularNucleus()
        
        # Higher brain areas (simplified)
        self.cortical_state = np.zeros(100)  # Cortical activity
        
        self.logger.info("Virtual Brain initialized with all structures")
    
    def process(
        self,
        sensory_input: Dict[str, Any],
        physiological_state: Dict[str, Any],
        emotional_state: Dict[str, Any],
        delta_time: float
    ) -> Dict[str, Any]:
        """
        Process all brain structures for one time step.
        
        This is where the virtual brain generates all behaviors in real time.
        
        Args:
            sensory_input: Sensory inputs
            physiological_state: Current physiological state
            emotional_state: Current emotional state
            delta_time: Time step
            
        Returns:
            Complete brain state and outputs
        """
        # Process paraventricular nucleus
        pvn_signals = self.paraventricular_nucleus.process(
            emotional_state, physiological_state, delta_time
        )
        
        # Process pituitary gland
        hypothalamic_input = {
            'stress_level': physiological_state.get('stress_level', 0.0),
            'social_stimulation': pvn_signals['oxytocin_release'],
        }
        hormone_levels = self.pituitary_gland.release_hormones(
            hypothalamic_input, delta_time
        )
        
        # Process brainstem
        higher_brain_input = {
            'stress_level': physiological_state.get('stress_level', 0.0),
            'arousal': physiological_state.get('arousal', 0.5),
            'motor_intent': self.cortical_state[:10],  # Motor commands from cortex
        }
        brainstem_output = self.brainstem.process(
            higher_brain_input, sensory_input, delta_time
        )
        
        return {
            'brainstem_output': brainstem_output,
            'hormone_levels': hormone_levels,
            'pvn_activity': pvn_signals,
            'cortical_state': self.cortical_state.copy(),
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current brain state."""
        return {
            'pituitary_oxytocin': self.pituitary_gland.oxytocin,
            'pituitary_acth': self.pituitary_gland.acth,
            'pvn_activity': self.paraventricular_nucleus.activity,
            'cortical_mean_activity': float(np.mean(np.abs(self.cortical_state))),
        }
