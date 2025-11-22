"""
Brain Structures Module

Implements simplified models of key brain structures:
- Brain Stem: Controls basic life functions
- Pituitary Gland: Hormone regulation (oxytocin)
- Paraventricular Nucleus: Stress response and social behavior
"""

import numpy as np
from typing import Dict, Optional


class BrainStem:
    """
    Simplified brain stem model.
    Controls basic life functions and arousal levels.
    """
    
    def __init__(self):
        """Initialize brain stem with default state."""
        self.arousal_level = 0.5  # 0.0 (sleep) to 1.0 (high alert)
        self.respiratory_rate = 12.0  # breaths per minute
        self.heart_rate = 70.0  # beats per minute
        
    def update(self, sensory_input: float, stress_level: float) -> Dict[str, float]:
        """
        Update brain stem state based on inputs.
        
        Args:
            sensory_input: Intensity of sensory stimulation (0-1)
            stress_level: Current stress level (0-1)
            
        Returns:
            Dictionary of physiological states
        """
        # Update arousal based on sensory input and stress
        self.arousal_level = np.clip(
            0.5 + 0.3 * sensory_input + 0.2 * stress_level,
            0.0, 1.0
        )
        
        # Arousal affects heart rate and respiration
        self.heart_rate = 60.0 + 40.0 * self.arousal_level
        self.respiratory_rate = 10.0 + 10.0 * self.arousal_level
        
        return {
            'arousal': self.arousal_level,
            'heart_rate': self.heart_rate,
            'respiratory_rate': self.respiratory_rate
        }


class PituitaryGland:
    """
    Simplified pituitary gland model.
    Regulates oxytocin production for social bonding and emotion.
    """
    
    def __init__(self):
        """Initialize pituitary gland state."""
        self.oxytocin_level = 0.5  # Baseline oxytocin (0-1)
        self.baseline_oxytocin = 0.5
        
    def release_oxytocin(self, social_stimulus: float, positive_emotion: float) -> float:
        """
        Release oxytocin in response to social and emotional stimuli.
        
        Args:
            social_stimulus: Intensity of social interaction (0-1)
            positive_emotion: Level of positive emotion (0-1)
            
        Returns:
            Current oxytocin level
        """
        # Oxytocin increases with social bonding and positive emotions
        release = 0.3 * social_stimulus + 0.2 * positive_emotion
        
        # Update oxytocin with decay back to baseline
        self.oxytocin_level += release
        self.oxytocin_level = 0.9 * self.oxytocin_level + 0.1 * self.baseline_oxytocin
        self.oxytocin_level = np.clip(self.oxytocin_level, 0.0, 1.0)
        
        return self.oxytocin_level
    
    def get_state(self) -> Dict[str, float]:
        """Get current hormonal state."""
        return {
            'oxytocin': self.oxytocin_level
        }


class ParaventricularNucleus:
    """
    Simplified paraventricular nucleus (PVN) model.
    Regulates stress response and integrates with pituitary for hormone control.
    """
    
    def __init__(self):
        """Initialize PVN state."""
        self.stress_response = 0.0  # Current stress response level (0-1)
        self.cortisol_signal = 0.0  # Signal for cortisol release (0-1)
        
    def process_stress(self, threat_level: float, arousal: float) -> Dict[str, float]:
        """
        Process stress signals and generate appropriate response.
        
        Args:
            threat_level: Perceived threat level (0-1)
            arousal: Arousal level from brain stem (0-1)
            
        Returns:
            Dictionary of stress-related signals
        """
        # Compute stress response based on threat and arousal
        self.stress_response = np.clip(
            0.4 * threat_level + 0.3 * arousal,
            0.0, 1.0
        )
        
        # Generate cortisol release signal
        self.cortisol_signal = self.stress_response
        
        return {
            'stress_response': self.stress_response,
            'cortisol_signal': self.cortisol_signal
        }
    
    def modulate_oxytocin(self, oxytocin_level: float) -> float:
        """
        Modulate stress response based on oxytocin (social buffering).
        
        Args:
            oxytocin_level: Current oxytocin level (0-1)
            
        Returns:
            Modulated stress response
        """
        # Oxytocin reduces stress response (social buffering effect)
        modulated_stress = self.stress_response * (1.0 - 0.3 * oxytocin_level)
        return np.clip(modulated_stress, 0.0, 1.0)


class VirtualBrain:
    """
    Integrates all brain structures into a cohesive virtual brain.
    """
    
    def __init__(self):
        """Initialize all brain structures."""
        self.brain_stem = BrainStem()
        self.pituitary = PituitaryGland()
        self.pvn = ParaventricularNucleus()
        
        # Current state
        self.current_state: Dict[str, float] = {}
    
    def process(
        self,
        sensory_input: float = 0.0,
        social_stimulus: float = 0.0,
        threat_level: float = 0.0,
        positive_emotion: float = 0.0
    ) -> Dict[str, float]:
        """
        Process inputs through all brain structures.
        
        Args:
            sensory_input: Sensory stimulation intensity (0-1)
            social_stimulus: Social interaction intensity (0-1)
            threat_level: Perceived threat level (0-1)
            positive_emotion: Positive emotion level (0-1)
            
        Returns:
            Integrated brain state
        """
        # Process through PVN first to get stress response
        stress_signals = self.pvn.process_stress(threat_level, 0.5)
        
        # Update brain stem with sensory input and stress
        brainstem_state = self.brain_stem.update(
            sensory_input,
            stress_signals['stress_response']
        )
        
        # Pituitary releases oxytocin based on social and emotional input
        oxytocin = self.pituitary.release_oxytocin(social_stimulus, positive_emotion)
        
        # Oxytocin modulates stress (social buffering)
        modulated_stress = self.pvn.modulate_oxytocin(oxytocin)
        
        # Combine all states
        self.current_state = {
            **brainstem_state,
            **self.pituitary.get_state(),
            'stress_response': modulated_stress,
            'cortisol_signal': stress_signals['cortisol_signal']
        }
        
        return self.current_state
    
    def get_state(self) -> Dict[str, float]:
        """Get current complete brain state."""
        return self.current_state.copy()
