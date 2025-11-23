"""
Animated Person Visualization

This module provides a visual representation of a person/character that responds
to the cognitive system's neural network states in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
from typing import Dict, Any, Optional
import logging


class AnimatedPerson:
    """
    An animated visual representation of a person connected to the neural network.
    
    The character's appearance and movements are driven by the cognitive system's
    state including arousal, stress, emotions, heart rate, and other physiological
    and cognitive parameters.
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the animated person visualization.
        
        Args:
            figsize: Size of the figure (width, height)
        """
        self.logger = logging.getLogger(__name__)
        
        # Create figure and axes
        self.fig, self.axes = plt.subplots(1, 2, figsize=figsize)
        self.ax_person = self.axes[0]
        self.ax_metrics = self.axes[1]
        
        # Person state
        self.arousal = 0.5
        self.stress = 0.5
        self.mood = 0.5
        self.heart_rate = 70.0
        self.emotional_valence = 0.0
        self.emotional_arousal = 0.5
        self.attention = 0.5
        self.oxytocin = 0.5
        
        # Animation state
        self.time = 0.0
        self.heartbeat_phase = 0.0
        
        # Visual elements
        self.person_elements = {}
        self.metric_elements = {}
        
        # Setup the visualization
        self._setup_person_visualization()
        self._setup_metrics_visualization()
        
        self.logger.info("AnimatedPerson initialized")
    
    def _setup_person_visualization(self):
        """Setup the person character visualization."""
        self.ax_person.set_xlim(-2, 2)
        self.ax_person.set_ylim(-3, 3)
        self.ax_person.set_aspect('equal')
        self.ax_person.axis('off')
        self.ax_person.set_title('Animated Person - Neural Network Driven', 
                                  fontsize=14, fontweight='bold')
        
        # Create person elements (simplified stick figure with emotion representation)
        # Head (circle)
        self.person_elements['head'] = Circle((0, 1.5), 0.4, 
                                               facecolor='lightblue', 
                                               edgecolor='black', 
                                               linewidth=2)
        self.ax_person.add_patch(self.person_elements['head'])
        
        # Body (rectangle)
        self.person_elements['body'] = Rectangle((-0.3, 0), 0.6, 1.2, 
                                                  facecolor='lightgreen', 
                                                  edgecolor='black', 
                                                  linewidth=2)
        self.ax_person.add_patch(self.person_elements['body'])
        
        # Arms (lines will be drawn in update)
        self.person_elements['left_arm'], = self.ax_person.plot([], [], 'k-', linewidth=4)
        self.person_elements['right_arm'], = self.ax_person.plot([], [], 'k-', linewidth=4)
        
        # Legs (lines will be drawn in update)
        self.person_elements['left_leg'], = self.ax_person.plot([], [], 'k-', linewidth=4)
        self.person_elements['right_leg'], = self.ax_person.plot([], [], 'k-', linewidth=4)
        
        # Heartbeat indicator (pulsing circle around body)
        self.person_elements['heart_pulse'] = Circle((0, 0.6), 0.15, 
                                                      facecolor='red', 
                                                      alpha=0.5)
        self.ax_person.add_patch(self.person_elements['heart_pulse'])
        
        # State text
        self.person_elements['state_text'] = self.ax_person.text(
            0, -2.5, '', ha='center', fontsize=10, style='italic'
        )
    
    def _setup_metrics_visualization(self):
        """Setup the metrics display panel."""
        self.ax_metrics.set_xlim(0, 1)
        self.ax_metrics.set_ylim(0, 10)
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Neural Network State', 
                                   fontsize=14, fontweight='bold')
        
        # Create metric bars
        metrics = [
            ('Heart Rate', 9),
            ('Arousal', 8),
            ('Stress', 7),
            ('Mood', 6),
            ('Attention', 5),
            ('Oxytocin', 4),
            ('Emotion (Valence)', 3),
            ('Emotion (Arousal)', 2),
        ]
        
        for metric_name, y in metrics:
            # Label
            self.ax_metrics.text(0.02, y + 0.3, metric_name, 
                                fontsize=9, verticalalignment='center')
            
            # Background bar
            bg_bar = Rectangle((0.35, y), 0.6, 0.6, 
                              facecolor='lightgray', 
                              edgecolor='black', 
                              linewidth=1)
            self.ax_metrics.add_patch(bg_bar)
            
            # Value bar (will be updated)
            value_bar = Rectangle((0.35, y), 0.0, 0.6, 
                                 facecolor='blue', 
                                 alpha=0.7)
            self.ax_metrics.add_patch(value_bar)
            self.metric_elements[metric_name] = value_bar
            
            # Value text
            value_text = self.ax_metrics.text(0.98, y + 0.3, '0', 
                                             fontsize=9, 
                                             ha='right',
                                             verticalalignment='center')
            self.metric_elements[f'{metric_name}_text'] = value_text
    
    def update_state(self, state: Dict[str, Any]):
        """
        Update the person's state from the cognitive system.
        
        Args:
            state: Dictionary containing cognitive system state including:
                - arousal: Arousal level (0-1)
                - stress: Stress level (0-1)
                - mood: Mood level (0-1)
                - heart_rate: Heart rate in BPM
                - emotional_valence: Emotional valence (-1 to 1)
                - emotional_arousal: Emotional arousal (0-1+)
                - attention: Attention level (0-1)
                - oxytocin: Oxytocin level (0-1)
        """
        self.arousal = state.get('arousal', 0.5)
        self.stress = state.get('stress', 0.5)
        self.mood = state.get('mood', 0.5)
        self.heart_rate = state.get('heart_rate', 70.0)
        self.emotional_valence = state.get('emotional_valence', 0.0)
        self.emotional_arousal = state.get('emotional_arousal', 0.5)
        self.attention = state.get('attention_level', 0.5)
        self.oxytocin = state.get('oxytocin', 0.5)
    
    def _get_emotion_color(self):
        """Get color based on emotional state."""
        # Map valence and arousal to color
        # Positive valence -> greenish, Negative -> reddish
        # High arousal -> more saturated
        
        if self.emotional_valence > 0:
            # Positive emotions (green-yellow spectrum)
            r = 0.3 + (1 - self.mood) * 0.5
            g = 0.6 + self.mood * 0.4
            b = 0.3
        else:
            # Negative emotions (red-orange spectrum)
            r = 0.8 + self.stress * 0.2
            g = 0.3 - self.stress * 0.2
            b = 0.3 - self.stress * 0.2
        
        # Ensure values are in valid range
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)
        
        return (r, g, b)
    
    def _update_person_appearance(self):
        """Update the person's visual appearance based on state."""
        # Update head color based on emotion
        emotion_color = self._get_emotion_color()
        self.person_elements['head'].set_facecolor(emotion_color)
        
        # Update body position based on arousal (breathing-like movement)
        breathing = 0.05 * np.sin(self.time * 2)
        self.person_elements['body'].set_height(1.2 + breathing * self.arousal)
        
        # Update heartbeat visualization
        # Heartbeat rate affects pulsing
        self.heartbeat_phase += self.heart_rate / 60.0 * 0.1
        pulse_size = 0.15 + 0.05 * np.sin(self.heartbeat_phase * 2 * np.pi)
        pulse_alpha = 0.3 + 0.4 * np.abs(np.sin(self.heartbeat_phase * 2 * np.pi))
        self.person_elements['heart_pulse'].set_radius(pulse_size)
        self.person_elements['heart_pulse'].set_alpha(pulse_alpha)
        
        # Update arm positions based on stress and arousal
        # High stress -> arms closer to body
        # High arousal -> more movement
        arm_movement = 0.2 * np.sin(self.time * 3) * self.arousal
        arm_spread = 0.8 - self.stress * 0.3
        
        # Left arm
        left_arm_x = [-0.3, -arm_spread + arm_movement]
        left_arm_y = [1.0, 0.2]
        self.person_elements['left_arm'].set_data(left_arm_x, left_arm_y)
        
        # Right arm
        right_arm_x = [0.3, arm_spread - arm_movement]
        right_arm_y = [1.0, 0.2]
        self.person_elements['right_arm'].set_data(right_arm_x, right_arm_y)
        
        # Update leg positions with walking-like motion based on arousal
        leg_movement = 0.3 * np.sin(self.time * 4) * self.arousal
        
        # Left leg
        left_leg_x = [-0.2, -0.3 + leg_movement]
        left_leg_y = [0, -1.2]
        self.person_elements['left_leg'].set_data(left_leg_x, left_leg_y)
        
        # Right leg
        right_leg_x = [0.2, 0.3 - leg_movement]
        right_leg_y = [0, -1.2]
        self.person_elements['right_leg'].set_data(right_leg_x, right_leg_y)
        
        # Update state description
        if self.stress > 0.7:
            state_str = "High Stress"
        elif self.arousal > 0.7:
            state_str = "Highly Aroused"
        elif self.mood > 0.7:
            state_str = "Positive Mood"
        elif self.emotional_valence < -0.5:
            state_str = "Negative Emotion"
        else:
            state_str = "Calm State"
        
        self.person_elements['state_text'].set_text(state_str)
    
    def _update_metrics_display(self):
        """Update the metrics visualization panel."""
        metrics_data = {
            'Heart Rate': self.heart_rate / 120.0,  # Normalize to 0-1 (assuming max 120 bpm)
            'Arousal': self.arousal,
            'Stress': self.stress,
            'Mood': self.mood,
            'Attention': self.attention,
            'Oxytocin': self.oxytocin,
            'Emotion (Valence)': (self.emotional_valence + 1) / 2,  # Map -1,1 to 0,1
            'Emotion (Arousal)': min(self.emotional_arousal, 1.0),
        }
        
        for metric_name, value in metrics_data.items():
            # Update bar width
            bar = self.metric_elements[metric_name]
            bar.set_width(value * 0.6)
            
            # Update bar color based on value
            if metric_name == 'Stress' and value > 0.7:
                bar.set_facecolor('red')
            elif metric_name == 'Mood' and value > 0.7:
                bar.set_facecolor('green')
            else:
                bar.set_facecolor('blue')
            
            # Update text value
            text = self.metric_elements[f'{metric_name}_text']
            if metric_name == 'Heart Rate':
                text.set_text(f'{self.heart_rate:.0f} bpm')
            elif metric_name == 'Emotion (Valence)':
                text.set_text(f'{self.emotional_valence:.2f}')
            else:
                text.set_text(f'{value:.2f}')
    
    def update(self, frame=0):
        """
        Update animation frame.
        
        Args:
            frame: Frame number (for animation)
        """
        self.time += 0.05  # Increment time
        
        self._update_person_appearance()
        self._update_metrics_display()
        
        return list(self.person_elements.values()) + list(self.metric_elements.values())
    
    def show(self):
        """Display the visualization (static)."""
        self.update()
        plt.tight_layout()
        plt.show()
    
    def animate(self, interval=50, frames=200):
        """
        Create an animated visualization.
        
        Args:
            interval: Delay between frames in milliseconds
            frames: Number of frames to animate
            
        Returns:
            FuncAnimation object
        """
        anim = FuncAnimation(
            self.fig, 
            self.update, 
            frames=frames,
            interval=interval,
            blit=False
        )
        return anim
    
    def save_animation(self, filename, interval=50, frames=200):
        """
        Save animation to file.
        
        Args:
            filename: Output filename (e.g., 'animation.gif' or 'animation.mp4')
            interval: Delay between frames in milliseconds
            frames: Number of frames to animate
        """
        anim = self.animate(interval=interval, frames=frames)
        anim.save(filename, writer='pillow' if filename.endswith('.gif') else 'ffmpeg')
        self.logger.info(f"Animation saved to {filename}")
