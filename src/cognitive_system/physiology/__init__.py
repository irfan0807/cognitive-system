"""
Virtual Physiology and Nervous System

Implements a virtual nervous system and physiology that drives cognition.
This module connects biological processes to cognitive functions.
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class NervousSystemState:
    """Represents the current state of the nervous system."""
    sympathetic_activity: float  # Sympathetic (fight/flight) activity (0-1)
    parasympathetic_activity: float  # Parasympathetic (rest/digest) activity (0-1)
    neurotransmitter_levels: Dict[str, float]  # Levels of various neurotransmitters
    sensory_integration: np.ndarray  # Integrated sensory information


class Neurotransmitter:
    """Models neurotransmitter levels and their effects."""
    
    def __init__(self, name: str, baseline: float = 0.5, decay_rate: float = 0.1):
        """
        Initialize a neurotransmitter.
        
        Args:
            name: Name of the neurotransmitter
            baseline: Baseline level (0-1)
            decay_rate: Rate of decay back to baseline
        """
        self.name = name
        self.baseline = baseline
        self.level = baseline
        self.decay_rate = decay_rate
    
    def release(self, amount: float):
        """Release neurotransmitter."""
        self.level = np.clip(self.level + amount, 0.0, 1.0)
    
    def update(self):
        """Update neurotransmitter level (decay toward baseline)."""
        self.level = self.level * (1 - self.decay_rate) + self.baseline * self.decay_rate
        self.level = np.clip(self.level, 0.0, 1.0)


class AutonomicNervousSystem:
    """
    Models the autonomic nervous system (sympathetic and parasympathetic).
    Controls involuntary physiological responses.
    """
    
    def __init__(self):
        """Initialize the autonomic nervous system."""
        self.sympathetic = 0.3  # Baseline sympathetic activity
        self.parasympathetic = 0.7  # Baseline parasympathetic activity
    
    def process(self, stress: float, relaxation: float) -> Dict[str, float]:
        """
        Update autonomic balance based on stress and relaxation signals.
        
        Args:
            stress: Stress level (0-1)
            relaxation: Relaxation level (0-1)
            
        Returns:
            Autonomic state
        """
        # Stress activates sympathetic
        self.sympathetic = np.clip(0.3 + 0.6 * stress, 0.0, 1.0)
        
        # Relaxation (e.g., from oxytocin) activates parasympathetic
        self.parasympathetic = np.clip(0.3 + 0.6 * relaxation, 0.0, 1.0)
        
        # They are somewhat antagonistic
        total = self.sympathetic + self.parasympathetic
        if total > 0:
            self.sympathetic = self.sympathetic / total
            self.parasympathetic = self.parasympathetic / total
        
        return {
            'sympathetic': self.sympathetic,
            'parasympathetic': self.parasympathetic
        }


class VirtualNervousSystem:
    """
    Complete virtual nervous system integrating sensory, motor, and autonomic functions.
    """
    
    def __init__(self, sensory_dim: int = 128):
        """
        Initialize the virtual nervous system.
        
        Args:
            sensory_dim: Dimensionality of sensory integration
        """
        self.sensory_dim = sensory_dim
        
        # Autonomic nervous system
        self.autonomic = AutonomicNervousSystem()
        
        # Neurotransmitters
        self.neurotransmitters = {
            'dopamine': Neurotransmitter('dopamine', baseline=0.5, decay_rate=0.15),
            'serotonin': Neurotransmitter('serotonin', baseline=0.6, decay_rate=0.1),
            'norepinephrine': Neurotransmitter('norepinephrine', baseline=0.4, decay_rate=0.2),
            'gaba': Neurotransmitter('gaba', baseline=0.5, decay_rate=0.12),
        }
        
        # Sensory integration buffer
        self.sensory_buffer = np.zeros(sensory_dim)
    
    def integrate_sensory_input(
        self,
        visual: np.ndarray,
        auditory: np.ndarray,
        proprioceptive: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Integrate multimodal sensory inputs.
        
        Args:
            visual: Visual sensory input
            auditory: Auditory sensory input
            proprioceptive: Body position/movement sense
            
        Returns:
            Integrated sensory representation
        """
        # Combine different sensory modalities
        inputs = []
        
        # Visual input (first third of buffer)
        visual_dim = self.sensory_dim // 3
        if len(visual) >= visual_dim:
            inputs.append(visual[:visual_dim])
        else:
            padded = np.zeros(visual_dim)
            padded[:len(visual)] = visual
            inputs.append(padded)
        
        # Auditory input (second third)
        auditory_dim = self.sensory_dim // 3
        if len(auditory) >= auditory_dim:
            inputs.append(auditory[:auditory_dim])
        else:
            padded = np.zeros(auditory_dim)
            padded[:len(auditory)] = auditory
            inputs.append(padded)
        
        # Proprioceptive (final third)
        proprio_dim = self.sensory_dim - visual_dim - auditory_dim
        if proprioceptive is not None and len(proprioceptive) >= proprio_dim:
            inputs.append(proprioceptive[:proprio_dim])
        else:
            inputs.append(np.zeros(proprio_dim))
        
        self.sensory_buffer = np.concatenate(inputs)
        return self.sensory_buffer
    
    def modulate_neurotransmitters(
        self,
        reward: float = 0.0,
        stress: float = 0.0,
        social_bonding: float = 0.0
    ):
        """
        Modulate neurotransmitter levels based on experiences.
        
        Args:
            reward: Reward signal (0-1)
            stress: Stress signal (0-1)
            social_bonding: Social bonding signal (0-1)
        """
        # Reward increases dopamine
        if reward > 0:
            self.neurotransmitters['dopamine'].release(reward * 0.3)
        
        # Stress increases norepinephrine, decreases serotonin
        if stress > 0:
            self.neurotransmitters['norepinephrine'].release(stress * 0.4)
            self.neurotransmitters['serotonin'].level *= (1.0 - stress * 0.2)
        
        # Social bonding increases serotonin
        if social_bonding > 0:
            self.neurotransmitters['serotonin'].release(social_bonding * 0.3)
        
        # Update all neurotransmitters
        for nt in self.neurotransmitters.values():
            nt.update()
    
    def update(
        self,
        brain_state: Dict[str, float],
        reward: float = 0.0
    ) -> NervousSystemState:
        """
        Update the complete nervous system state.
        
        Args:
            brain_state: Current brain state from VirtualBrain
            reward: Reward signal
            
        Returns:
            Current nervous system state
        """
        # Extract relevant signals from brain state
        stress = brain_state.get('stress_response', 0.0)
        oxytocin = brain_state.get('oxytocin', 0.5)
        
        # Update autonomic nervous system
        autonomic_state = self.autonomic.process(stress, oxytocin)
        
        # Modulate neurotransmitters
        self.modulate_neurotransmitters(
            reward=reward,
            stress=stress,
            social_bonding=oxytocin
        )
        
        # Create state snapshot
        state = NervousSystemState(
            sympathetic_activity=autonomic_state['sympathetic'],
            parasympathetic_activity=autonomic_state['parasympathetic'],
            neurotransmitter_levels={
                name: nt.level for name, nt in self.neurotransmitters.items()
            },
            sensory_integration=self.sensory_buffer.copy()
        )
        
        return state
    
    def get_cognitive_modulation(self) -> Dict[str, float]:
        """
        Get how neurotransmitters modulate cognitive functions.
        
        Returns:
            Dictionary of cognitive modulation factors
        """
        return {
            'attention': self.neurotransmitters['norepinephrine'].level,
            'mood': self.neurotransmitters['serotonin'].level,
            'motivation': self.neurotransmitters['dopamine'].level,
            'calmness': self.neurotransmitters['gaba'].level,
        }


class VirtualPhysiology:
    """
    Complete virtual physiology system.
    Integrates nervous system with other physiological processes.
    """
    
    def __init__(self):
        """Initialize virtual physiology."""
        self.nervous_system = VirtualNervousSystem()
        
        # Metabolic state
        self.energy_level = 1.0  # Energy available (0-1)
        self.homeostatic_balance = 0.5  # How balanced internal state is (0-1)
    
    def update(
        self,
        brain_state: Dict[str, float],
        sensory_inputs: Dict[str, np.ndarray],
        reward: float = 0.0
    ) -> Dict[str, any]:
        """
        Update complete physiological state.
        
        Args:
            brain_state: Current brain state
            sensory_inputs: Dictionary of sensory inputs (visual, auditory, etc.)
            reward: Reward signal
            
        Returns:
            Complete physiological state
        """
        # Integrate sensory information
        visual = sensory_inputs.get('visual', np.zeros(32))
        auditory = sensory_inputs.get('auditory', np.zeros(32))
        proprio = sensory_inputs.get('proprioceptive', None)
        
        self.nervous_system.integrate_sensory_input(visual, auditory, proprio)
        
        # Update nervous system
        ns_state = self.nervous_system.update(brain_state, reward)
        
        # Energy expenditure based on activity
        arousal = brain_state.get('arousal', 0.5)
        energy_cost = 0.01 * arousal
        self.energy_level = np.clip(self.energy_level - energy_cost, 0.0, 1.0)
        
        # Homeostatic balance affected by stress and autonomic balance
        stress = brain_state.get('stress_response', 0.0)
        self.homeostatic_balance = 0.9 * self.homeostatic_balance + 0.1 * (
            1.0 - stress + ns_state.parasympathetic_activity
        ) / 2.0
        self.homeostatic_balance = np.clip(self.homeostatic_balance, 0.0, 1.0)
        
        return {
            'nervous_system_state': ns_state,
            'energy_level': self.energy_level,
            'homeostatic_balance': self.homeostatic_balance,
            'cognitive_modulation': self.nervous_system.get_cognitive_modulation()
        }
    
    def rest(self, duration: float = 1.0):
        """
        Simulate rest/recovery period.
        
        Args:
            duration: Duration of rest (arbitrary units)
        """
        # Restore energy
        self.energy_level = np.clip(self.energy_level + 0.1 * duration, 0.0, 1.0)
        
        # Improve homeostatic balance
        self.homeostatic_balance = np.clip(
            self.homeostatic_balance + 0.05 * duration,
            0.0, 1.0
        )
