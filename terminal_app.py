#!/usr/bin/env python3
"""
Cognitive System - Terminal Interface

Interactive terminal application where the AI can see you through the camera
and talk to you through text-to-speech with a retro green terminal theme.

Usage:
    python terminal_app.py              # With camera and audio
    python terminal_app.py --no-camera  # Text-only mode
    python terminal_app.py --no-speech  # Silent mode
"""

import sys
import os
import time
import logging
import argparse
import random
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich import box

# Import cognitive system components
from cognitive_system.core.embodied_cognition import EmbodiedCognitionSystem
from cognitive_system.virtual_biology.nervous_system import VirtualNervousSystem
from cognitive_system.core.neural_network import NeuralNetworkController
from cognitive_system.memory.multimodal_memory import MultimodalMemorySystem
from cognitive_system.behavior.behavior_engine import BehaviorEngine
from cognitive_system.utils.terminal_video import TerminalVideoDisplay
from cognitive_system.utils.tts_engine import TextToSpeechEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/tmp/cognitive_terminal.log')]
)
logger = logging.getLogger(__name__)


class TerminalCognitiveApp:
    """
    Terminal-based cognitive system interface.
    """
    
    def __init__(self, use_camera: bool = True, use_speech: bool = True):
        """
        Initialize terminal cognitive app.
        
        Args:
            use_camera: Enable camera input
            use_speech: Enable text-to-speech output
        """
        # Create console with green theme
        self.console = Console(
            color_system="truecolor",
            force_terminal=True,
            force_interactive=True
        )
        
        self.use_camera = use_camera
        self.use_speech = use_speech
        
        # Initialize components
        self.cognitive_system = None
        self.nervous_system = None
        self.neural_network = None
        self.memory_system = None
        self.behavior_engine = None
        self.tts_engine = None
        self.video_display = None
        self.camera = None
        
        # State
        self.running = False
        self.frame_count = 0
        self.conversation_history = []
        self.current_state = {}
        
    def initialize(self):
        """Initialize all system components."""
        self.console.clear()
        self.console.print(Panel.fit(
            "[bold green]COGNITIVE SYSTEM - TERMINAL INTERFACE[/bold green]\n"
            "[green]Initializing...[/green]",
            border_style="green"
        ))
        
        # Initialize cognitive system
        self.console.print("[green]→[/green] Initializing cognitive system...")
        self.cognitive_system = EmbodiedCognitionSystem()
        self.nervous_system = VirtualNervousSystem()
        self.neural_network = NeuralNetworkController(
            input_size=22,
            hidden_size=128,
            output_size=32
        )
        self.memory_system = MultimodalMemorySystem()
        self.behavior_engine = BehaviorEngine()
        
        self.cognitive_system.setup(
            virtual_biology=self.nervous_system,
            neural_network=self.neural_network,
            memory_system=self.memory_system,
            behavior_engine=self.behavior_engine
        )
        self.cognitive_system.start()
        self.console.print("[green]✓[/green] Cognitive system ready")
        
        # Initialize TTS
        if self.use_speech:
            self.console.print("[green]→[/green] Initializing speech engine...")
            self.tts_engine = TextToSpeechEngine(rate=150, volume=0.9)
            self.tts_engine.start()
            self.console.print("[green]✓[/green] Speech engine ready")
        
        # Initialize camera
        if self.use_camera:
            self.console.print("[green]→[/green] Initializing camera...")
            try:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FPS, 15)
                    self.video_display = TerminalVideoDisplay(width=60, height=20)
                    self.console.print("[green]✓[/green] Camera ready")
                else:
                    self.console.print("[yellow]⚠[/yellow] Camera not available")
                    self.use_camera = False
            except Exception as e:
                self.console.print(f"[yellow]⚠[/yellow] Camera error: {e}")
                self.use_camera = False
        
        time.sleep(1)
    
    def speak(self, text: str):
        """
        Make the system speak and show in terminal.
        
        Args:
            text: Text to speak
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_history.append({
            'time': timestamp,
            'speaker': 'AI',
            'text': text
        })
        
        if self.tts_engine:
            self.tts_engine.speak(text)
    
    def extract_visual_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract visual features from camera frame.
        
        Args:
            frame: OpenCV frame
            
        Returns:
            Visual feature vector
        """
        frame_resized = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray_normalized = gray.astype(np.float32) / 255.0
        visual_features = gray_normalized.flatten()[:64]
        
        if len(visual_features) < 64:
            visual_features = np.pad(visual_features, (0, 64 - len(visual_features)))
        
        return visual_features
    
    def create_layout(self, video_ascii: str = "", state_info: dict = None) -> Layout:
        """
        Create the terminal layout.
        
        Args:
            video_ascii: ASCII art video frame
            state_info: Current system state
            
        Returns:
            Rich Layout object
        """
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=8)
        )
        
        # Header
        header_text = Text()
        header_text.append("╔═══════════════════════════════════════════════════════════════════╗\n", style="bold green")
        header_text.append("║  ", style="green")
        header_text.append("COGNITIVE SYSTEM - LIVE TERMINAL INTERFACE", style="bold green")
        header_text.append("                  ║\n", style="green")
        header_text.append("╚═══════════════════════════════════════════════════════════════════╝", style="bold green")
        layout["header"].update(Panel(header_text, border_style="green", box=box.HEAVY))
        
        # Main area - split into video and state
        layout["main"].split_row(
            Layout(name="video", ratio=2),
            Layout(name="state", ratio=1)
        )
        
        # Video panel
        if video_ascii:
            video_panel = Panel(
                video_ascii,
                title="[bold green]📹 Camera Feed[/bold green]",
                border_style="green",
                box=box.HEAVY
            )
        else:
            video_panel = Panel(
                "[dim green]No camera feed[/dim green]",
                title="[bold green]📹 Camera Feed[/bold green]",
                border_style="green",
                box=box.HEAVY
            )
        layout["video"].update(video_panel)
        
        # State panel
        if state_info:
            state_table = Table(show_header=False, box=None, padding=(0, 1))
            state_table.add_column("Metric", style="green")
            state_table.add_column("Value", style="bold green")
            
            state_table.add_row("Arousal:", f"{state_info.get('arousal', 0.0):.2f}")
            state_table.add_row("Mood:", f"{state_info.get('mood', 0.0):.2f}")
            state_table.add_row("Stress:", f"{state_info.get('stress', 0.0):.2f}")
            state_table.add_row("Heart Rate:", f"{state_info.get('heart_rate', 60.0):.0f} bpm")
            state_table.add_row("Attention:", f"{state_info.get('attention', 0.0):.2f}")
            state_table.add_row("Frame:", f"{self.frame_count}")
            
            state_panel = Panel(
                state_table,
                title="[bold green]🧠 System State[/bold green]",
                border_style="green",
                box=box.HEAVY
            )
        else:
            state_panel = Panel(
                "[dim green]No state data[/dim green]",
                title="[bold green]🧠 System State[/bold green]",
                border_style="green",
                box=box.HEAVY
            )
        layout["state"].update(state_panel)
        
        # Footer - conversation
        conversation_text = ""
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            speaker_color = "bold green" if msg['speaker'] == 'AI' else "green"
            conversation_text += f"[{speaker_color}]{msg['time']} {msg['speaker']}:[/{speaker_color}] [green]{msg['text']}[/green]\n"
        
        if not conversation_text:
            conversation_text = "[dim green]Conversation will appear here...[/dim green]"
        
        footer_panel = Panel(
            conversation_text,
            title="[bold green]💬 Conversation[/bold green]",
            border_style="green",
            box=box.HEAVY
        )
        layout["footer"].update(footer_panel)
        
        return layout
    
    def run(self):
        """Run the terminal application."""
        self.running = True
        
        # Initial greeting
        greeting = "Hello! I am your cognitive system. I can see you through the camera and I'm here to interact with you."
        self.speak(greeting)
        
        try:
            with Live(self.create_layout(), console=self.console, refresh_per_second=4, screen=True) as live:
                last_speech_time = time.time()
                speech_interval = 5.0  # Speak every 5 seconds
                
                while self.running:
                    try:
                        # Get video frame
                        video_ascii = ""
                        visual_features = None
                        
                        if self.use_camera and self.camera and self.camera.isOpened():
                            ret, frame = self.camera.read()
                            if ret:
                                # Flip for mirror effect
                                frame = cv2.flip(frame, 1)
                                
                                # Convert to ASCII
                                video_ascii = self.video_display.frame_to_ascii(frame)
                                
                                # Extract features
                                visual_features = self.extract_visual_features(frame)
                        
                        # Use synthetic features if no camera
                        if visual_features is None:
                            visual_features = np.random.randn(64) * 0.1
                        
                        # Synthetic audio features
                        auditory_features = np.random.randn(32) * 0.05
                        
                        # Update cognitive system
                        sensory_input = {
                            'visual': visual_features[:10],
                            'auditory': auditory_features[:10],
                        }
                        self.cognitive_system.update(1.0/15, sensory_input)
                        
                        # Get system state
                        state = self.nervous_system.get_state()
                        
                        # Update current state
                        self.current_state = {
                            'arousal': state.get('arousal', 0.5),
                            'stress': state.get('stress_level', 0.0),
                            'mood': state.get('mood', 0.5),
                            'heart_rate': state.get('heart_rate', 60.0),
                            'attention': state.get('attention', 0.5),
                        }
                        
                        # Periodic AI responses based on state
                        current_time = time.time()
                        if current_time - last_speech_time > speech_interval:
                            # Generate context-aware response
                            response = self.generate_response(self.current_state)
                            if response:
                                self.speak(response)
                            last_speech_time = current_time
                        
                        # Update display
                        layout = self.create_layout(video_ascii, self.current_state)
                        live.update(layout)
                        
                        self.frame_count += 1
                        time.sleep(1.0 / 4)  # 4 FPS for terminal
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}", exc_info=True)
                        time.sleep(0.1)
        
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def generate_response(self, state: dict) -> str:
        """
        Generate AI response based on system state.
        
        Args:
            state: Current system state
            
        Returns:
            Response text
        """
        arousal = state.get('arousal', 0.5)
        mood = state.get('mood', 0.5)
        stress = state.get('stress', 0.0)
        
        responses = []
        
        # Arousal-based responses
        if arousal > 0.7:
            responses.append("I'm feeling quite alert and focused right now.")
        elif arousal < 0.3:
            responses.append("I'm in a calm and relaxed state.")
        
        # Mood-based responses
        if mood > 0.6:
            responses.append("I'm experiencing positive emotions.")
        elif mood < 0.4:
            responses.append("My mood is somewhat neutral at the moment.")
        
        # Stress-based responses
        if stress > 0.5:
            responses.append("I'm detecting some elevated stress levels.")
        
        # Observation responses
        if self.use_camera:
            responses.append("I can see you through the camera.")
        
        # Return a random response or None
        if responses and random.random() < 0.7:  # 70% chance to speak
            return random.choice(responses)
        return None
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        self.console.print("\n[green]Shutting down...[/green]")
        
        if self.cognitive_system:
            self.cognitive_system.stop()
        
        if self.tts_engine:
            self.tts_engine.stop()
        
        if self.camera:
            self.camera.release()
        
        self.console.print("[green]✓ Goodbye![/green]\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cognitive System Terminal Interface"
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable camera input"
    )
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="Disable text-to-speech output"
    )
    
    args = parser.parse_args()
    
    app = TerminalCognitiveApp(
        use_camera=not args.no_camera,
        use_speech=not args.no_speech
    )
    
    try:
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        app.cleanup()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
