"""
Virtual Nervous System

Implements the whole virtual nervous system and virtual physiology that drives
the character. The system generates physiological responses based on stimuli
and internal state.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging


class VirtualNervousSystem:
    """
    Virtual Nervous System controlling the character's physiology.
    
    The nervous system integrates with the virtual brain to control physical
    aspects and generate physiological responses (e.g., increased heart rate
    when stressed).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Virtual Nervous System.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Physiological parameters
        self.base_heart_rate = 60.0  # beats per minute
        self.heart_rate = self.base_heart_rate
        self.breathing_rate = 12.0  # breaths per minute
        self.stress_level = 0.0  # 0.0 to 1.0
        self.arousal = 0.5  # 0.0 to 1.0
        
        # Chemical regulation (simulated neurotransmitters/hormones)
        self.oxytocin_level = 0.5
        self.cortisol_level = 0.3
        self.dopamine_level = 0.5
        
        # Motor control state
        self.motor_state = np.zeros(10)  # 10 motor control parameters
        
        self.logger.info("Virtual Nervous System initialized")
    
    def process(
        self, 
        sensory_input: Dict[str, Any],
        delta_time: float
    ) -> Dict[str, Any]:
        """
        Process sensory input and update physiological state.
        
        Args:
            sensory_input: Sensory inputs from environment
            delta_time: Time step in seconds
            
        Returns:
            Current physiological state
        """
        # Update stress based on sensory input
        self._update_stress(sensory_input, delta_time)
        
        # Update heart rate based on stress
        self._update_heart_rate(delta_time)
        
        # Update breathing rate
        self._update_breathing(delta_time)
        
        # Update chemical levels
        self._update_chemical_regulation(delta_time)
        
        # Update arousal
        self._update_arousal(sensory_input, delta_time)
        
        return self.get_state()
    
    def _update_stress(self, sensory_input: Dict[str, Any], delta_time: float):
        """Update stress level based on sensory input."""
        # Simple stress model - increases with intense stimuli
        visual_intensity = self._get_stimulus_intensity(sensory_input.get('visual', 0))
        auditory_intensity = self._get_stimulus_intensity(sensory_input.get('auditory', 0))
        
        # Stress increases with stimulus intensity, decreases naturally
        stress_change = (visual_intensity + auditory_intensity) * 0.1 - self.stress_level * 0.05
        self.stress_level = np.clip(self.stress_level + stress_change * delta_time, 0.0, 1.0)
    
    def _update_heart_rate(self, delta_time: float):
        """
        Update heart rate based on stress level.
        
        If the character gets stressed, the virtual heart starts beating faster.
        """
        # Target heart rate increases with stress (60-120 bpm range)
        target_heart_rate = self.base_heart_rate + self.stress_level * 60.0
        
        # Smooth transition to target
        rate_change = (target_heart_rate - self.heart_rate) * 0.1
        self.heart_rate += rate_change * delta_time
    
    def _update_breathing(self, delta_time: float):
        """
        Update breathing rate based on stress and arousal.
        
        The virtual breathing starts beating faster under stress.
        """
        # Target breathing rate increases with stress and arousal (12-30 breaths/min)
        target_breathing_rate = 12.0 + (self.stress_level + self.arousal) * 9.0
        
        # Smooth transition
        breathing_change = (target_breathing_rate - self.breathing_rate) * 0.1
        self.breathing_rate += breathing_change * delta_time
    
    def _update_chemical_regulation(self, delta_time: float):
        """
        Update chemical regulation (neurotransmitters and hormones).
        
        Simulates glands like the pituitary releasing substances like oxytocin.
        """
        # Cortisol increases with stress
        cortisol_target = 0.3 + self.stress_level * 0.5
        self.cortisol_level += (cortisol_target - self.cortisol_level) * 0.05 * delta_time
        
        # Oxytocin decreases with stress, increases with positive stimuli
        oxytocin_target = 0.5 - self.stress_level * 0.3
        self.oxytocin_level += (oxytocin_target - self.oxytocin_level) * 0.05 * delta_time
        
        # Dopamine related to arousal
        dopamine_target = self.arousal
        self.dopamine_level += (dopamine_target - self.dopamine_level) * 0.05 * delta_time
        
        # Clamp values
        self.oxytocin_level = np.clip(self.oxytocin_level, 0.0, 1.0)
        self.cortisol_level = np.clip(self.cortisol_level, 0.0, 1.0)
        self.dopamine_level = np.clip(self.dopamine_level, 0.0, 1.0)
    
    def _update_arousal(self, sensory_input: Dict[str, Any], delta_time: float):
        """Update arousal level based on sensory input and internal state."""
        # Arousal increases with stimulus and stress
        visual_intensity = self._get_stimulus_intensity(sensory_input.get('visual', 0))
        auditory_intensity = self._get_stimulus_intensity(sensory_input.get('auditory', 0))
        
        target_arousal = (visual_intensity + auditory_intensity + self.stress_level) / 3.0
        arousal_change = (target_arousal - self.arousal) * 0.1
        self.arousal = np.clip(self.arousal + arousal_change * delta_time, 0.0, 1.0)
    
    def _get_stimulus_intensity(self, stimulus: Any) -> float:
        """Calculate intensity of a stimulus."""
        if isinstance(stimulus, (int, float)):
            return float(np.clip(stimulus, 0.0, 1.0))
        elif isinstance(stimulus, np.ndarray):
            return float(np.clip(np.mean(np.abs(stimulus)), 0.0, 1.0))
        elif isinstance(stimulus, (list, tuple)):
            return float(np.clip(np.mean(np.abs(np.array(stimulus))), 0.0, 1.0))
        return 0.0
    
    def respond_to_music(self, music_beat: float, delta_time: float):
        """
        Respond to music by automatically responding to the beat.
        
        The character's stimulus system is affected by music, generating
        corresponding movement patterns in the body.
        
        Args:
            music_beat: Beat strength (0.0 to 1.0)
            delta_time: Time step
        """
        # Generate rhythmic motor response
        beat_intensity = music_beat
        
        # Modulate motor state with beat
        for i in range(len(self.motor_state)):
            phase = (i / len(self.motor_state)) * 2 * np.pi
            self.motor_state[i] = beat_intensity * np.sin(
                phase + self.arousal * np.pi
            )
        
        # Increase arousal with music
        arousal_boost = beat_intensity * 0.1
        self.arousal = np.clip(self.arousal + arousal_boost * delta_time, 0.0, 1.0)
    
    def set_stress_level(self, stress: float):
        """
        Set the stress level (for simulation/testing purposes).
        
        Args:
            stress: Stress level (0.0 to 1.0)
        """
        self.stress_level = np.clip(stress, 0.0, 1.0)
    
    def simulate_emotional_response(self, valence: float, arousal: float, stress: float):
        """
        Simulate an emotional response by setting physiological parameters.
        
        Useful for testing and conditioning scenarios.
        
        Args:
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Arousal level (0.0 to 1.0)
            stress: Stress level (0.0 to 1.0)
        """
        self.stress_level = np.clip(stress, 0.0, 1.0)
        self.arousal = np.clip(arousal, 0.0, 1.0)
        
        # Adjust chemical levels based on valence
        if valence > 0:
            self.oxytocin_level = np.clip(0.5 + valence * 0.5, 0.0, 1.0)
            self.cortisol_level = np.clip(0.3 - valence * 0.3, 0.0, 1.0)
        else:
            self.oxytocin_level = np.clip(0.5 + valence * 0.5, 0.0, 1.0)
            self.cortisol_level = np.clip(0.3 - valence * 0.3, 0.0, 1.0)

    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current physiological state.
        
        Returns:
            Dictionary containing all physiological parameters
        """
        return {
            'heart_rate': self.heart_rate,
            'breathing_rate': self.breathing_rate,
            'stress_level': self.stress_level,
            'arousal': self.arousal,
            'chemical_levels': {
                'oxytocin': self.oxytocin_level,
                'cortisol': self.cortisol_level,
                'dopamine': self.dopamine_level,
            },
            'motor_state': self.motor_state.copy(),
            'emotional_state': {
                'valence': self.oxytocin_level - self.cortisol_level,  # Positive/negative
                'arousal': self.arousal,
            }
        }
