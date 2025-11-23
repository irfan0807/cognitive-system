"""
Stress Controller for Animation

Provides interactive controls to increase/decrease stress levels in the animation.
Supports keyboard input, API calls, and programmatic control.
"""

import logging
from typing import Callable, Optional, Dict, Any
from enum import Enum


class StressLevel(Enum):
    """Stress level presets."""
    VERY_LOW = 0.0
    LOW = 0.2
    NORMAL = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0


class StressController:
    """Controller for managing stress levels."""
    
    def __init__(self, initial_stress: float = 0.4, step_size: float = 0.1):
        """
        Initialize stress controller.
        
        Args:
            initial_stress: Starting stress level (0.0-1.0)
            step_size: Amount to increase/decrease per step
        """
        self.logger = logging.getLogger(__name__)
        self.current_stress = max(0.0, min(1.0, initial_stress))
        self.step_size = step_size
        self.min_stress = 0.0
        self.max_stress = 1.0
        
        # Callbacks for stress changes
        self.on_stress_change: Optional[Callable[[float], None]] = None
        
        self.logger.info(f"StressController initialized: stress={self.current_stress:.2f}, step={step_size}")
    
    def increase_stress(self, amount: Optional[float] = None) -> float:
        """
        Increase stress level.
        
        Args:
            amount: Amount to increase (default: step_size)
            
        Returns:
            New stress level
        """
        if amount is None:
            amount = self.step_size
        
        old_stress = self.current_stress
        self.current_stress = min(self.max_stress, self.current_stress + amount)
        
        self.logger.info(f"STRESS UP: {old_stress:.2f} → {self.current_stress:.2f}")
        
        if self.on_stress_change:
            self.on_stress_change(self.current_stress)
        
        return self.current_stress
    
    def decrease_stress(self, amount: Optional[float] = None) -> float:
        """
        Decrease stress level.
        
        Args:
            amount: Amount to decrease (default: step_size)
            
        Returns:
            New stress level
        """
        if amount is None:
            amount = self.step_size
        
        old_stress = self.current_stress
        self.current_stress = max(self.min_stress, self.current_stress - amount)
        
        self.logger.info(f"STRESS DOWN: {old_stress:.2f} → {self.current_stress:.2f}")
        
        if self.on_stress_change:
            self.on_stress_change(self.current_stress)
        
        return self.current_stress
    
    def set_stress(self, level: float) -> float:
        """
        Set stress to specific level.
        
        Args:
            level: Target stress level (0.0-1.0)
            
        Returns:
            New stress level
        """
        old_stress = self.current_stress
        self.current_stress = max(self.min_stress, min(self.max_stress, level))
        
        self.logger.info(f"STRESS SET: {old_stress:.2f} → {self.current_stress:.2f}")
        
        if self.on_stress_change:
            self.on_stress_change(self.current_stress)
        
        return self.current_stress
    
    def set_preset(self, preset: StressLevel) -> float:
        """
        Set stress to a preset level.
        
        Args:
            preset: Preset stress level
            
        Returns:
            New stress level
        """
        return self.set_stress(preset.value)
    
    def get_stress(self) -> float:
        """Get current stress level."""
        return self.current_stress
    
    def get_stress_description(self) -> str:
        """Get human-readable stress description."""
        stress = self.current_stress
        
        if stress <= 0.1:
            return "Very Relaxed"
        elif stress <= 0.3:
            return "Relaxed"
        elif stress <= 0.5:
            return "Normal"
        elif stress <= 0.7:
            return "Tense"
        elif stress <= 0.9:
            return "Very Tense"
        else:
            return "Extremely Stressed"
    
    def get_status_string(self) -> str:
        """Get formatted status string."""
        stress = self.current_stress
        bar_length = 20
        filled = int(bar_length * stress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        return f"[{bar}] {stress:.1%} - {self.get_stress_description()}"


class InteractiveStressController(StressController):
    """Stress controller with keyboard input support."""
    
    def __init__(self, *args, **kwargs):
        """Initialize interactive stress controller."""
        super().__init__(*args, **kwargs)
        self.keyboard_enabled = False
    
    def handle_key_press(self, key: str) -> Optional[float]:
        """
        Handle keyboard input.
        
        Args:
            key: Key pressed ('+', '-', 'r', '1'-'6', etc.)
            
        Returns:
            New stress level if changed, None otherwise
        """
        if key in ['+', '=']:
            return self.increase_stress()
        
        elif key in ['-', '_']:
            return self.decrease_stress()
        
        elif key.lower() == 'r':
            return self.set_stress(0.4)  # Reset to normal
        
        elif key == '0':
            return self.set_preset(StressLevel.VERY_LOW)
        elif key == '1':
            return self.set_preset(StressLevel.VERY_LOW)
        elif key == '2':
            return self.set_preset(StressLevel.LOW)
        elif key == '3':
            return self.set_preset(StressLevel.NORMAL)
        elif key == '4':
            return self.set_preset(StressLevel.MODERATE)
        elif key == '5':
            return self.set_preset(StressLevel.HIGH)
        elif key == '6':
            return self.set_preset(StressLevel.VERY_HIGH)
        
        return None
    
    def get_help_text(self) -> str:
        """Get keyboard help text."""
        return """
STRESS CONTROL KEYS:
  + or =           Increase stress
  - or _           Decrease stress
  0-1              Very Low stress
  2                Low stress
  3                Normal stress (default)
  4                Moderate stress
  5                High stress
  6                Very High stress
  R                Reset to normal
"""


class StressAnimationBridge:
    """Bridge stress controller with animation."""
    
    def __init__(self, stress_controller: StressController):
        """
        Initialize bridge.
        
        Args:
            stress_controller: StressController instance
        """
        self.stress_controller = stress_controller
        self.logger = logging.getLogger(__name__)
    
    def get_animation_state_with_stress(self, base_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify animation state based on stress level.
        
        Args:
            base_state: Base animation state dictionary
            
        Returns:
            Modified state with stress applied
        """
        stress = self.stress_controller.get_stress()
        
        # Modify state based on stress
        modified_state = base_state.copy()
        modified_state['stress'] = stress
        
        # Stress affects arousal
        if 'arousal' in modified_state:
            modified_state['arousal'] = 0.3 + 0.7 * stress
        
        # Stress affects heart rate
        if 'heart_rate' in modified_state:
            base_hr = 60.0
            modified_state['heart_rate'] = base_hr + 40 * stress
        
        # Stress affects mood (negative impact)
        if 'mood' in modified_state:
            modified_state['mood'] = max(0.0, modified_state['mood'] - 0.3 * stress)
        
        # Stress increases attention (focusing on threat)
        if 'attention' in modified_state:
            modified_state['attention'] = min(1.0, modified_state['attention'] + 0.2 * stress)
        
        return modified_state
    
    def print_controls(self) -> None:
        """Print control information to console."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("STRESS CONTROL - KEYBOARD SHORTCUTS")
        self.logger.info("=" * 70)
        
        if isinstance(self.stress_controller, InteractiveStressController):
            self.logger.info(self.stress_controller.get_help_text())
        
        self.logger.info("=" * 70)
        self.logger.info("")
