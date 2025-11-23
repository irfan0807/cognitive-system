#!/usr/bin/env python3
"""
Cognitive System - Terminal Interface with RAG Integration

Interactive terminal application where the AI can:
- See you through the camera (ASCII art)
- Hear and respond to you (text-to-speech)
- Chat with you (interactive input)
- Make decisions using RAG (Retrieval-Augmented Generation)

Usage:
    python terminal_app_rag.py              # Full mode (camera, audio, RAG, chat)
    python terminal_app_rag.py --no-camera  # Text-only mode
    python terminal_app_rag.py --no-speech  # Silent mode (no audio)
    python terminal_app_rag.py --no-rag     # Without RAG system
"""

import sys
import os
import time
import logging
import argparse
import threading
import queue
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
from cognitive_system.rag.multimodal_rag import MultimodalRAGSystem
from cognitive_system.utils.terminal_video import TerminalVideoDisplay
from cognitive_system.utils.tts_engine import TextToSpeechEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/tmp/cognitive_terminal_rag.log')]
)
logger = logging.getLogger(__name__)


class TerminalCognitiveAppWithRAG:
    """
    Terminal-based cognitive system interface with RAG integration and user interaction.
    """
    
    def __init__(self, use_camera: bool = True, use_speech: bool = True, use_rag: bool = True):
        """
        Initialize terminal cognitive app.
        
        Args:
            use_camera: Enable camera input
            use_speech: Enable text-to-speech output
            use_rag: Enable RAG system for decision-making
        """
        # Create console with green theme
        self.console = Console(
            color_system="truecolor",
            force_terminal=True,
            force_interactive=True
        )
        
        self.use_camera = use_camera
        self.use_speech = use_speech
        self.use_rag = use_rag
        
        # Initialize components
        self.cognitive_system = None
        self.nervous_system = None
        self.neural_network = None
        self.memory_system = None
        self.behavior_engine = None
        self.rag_system = None
        self.tts_engine = None
        self.video_display = None
        self.camera = None
        
        # State
        self.running = False
        self.frame_count = 0
        self.conversation_history = []
        self.current_state = {}
        self.user_input_queue = queue.Queue()
        self.rag_decision_log = []
        
    def initialize(self):
        """Initialize all system components."""
        self.console.clear()
        self.console.print(Panel.fit(
            "[bold green]COGNITIVE SYSTEM - TERMINAL WITH RAG[/bold green]\n"
            "[green]Initializing system components...[/green]",
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
        
        # Initialize RAG system
        if self.use_rag:
            self.console.print("[green]→[/green] Initializing RAG system...")
            try:
                self.rag_system = MultimodalRAGSystem(embedding_dim=128)
                self.rag_system.attach_cognitive_system(self.cognitive_system)
                self.rag_system.start()
                self.console.print("[green]✓[/green] RAG system ready")
            except Exception as e:
                self.console.print(f"[yellow]⚠[/yellow] RAG system initialization failed: {e}")
                self.use_rag = False
        
        # Initialize TTS
        if self.use_speech:
            self.console.print("[green]→[/green] Initializing speech engine...")
            try:
                self.tts_engine = TextToSpeechEngine(rate=150, volume=0.9)
                self.tts_engine.start()
                self.console.print("[green]✓[/green] Speech engine ready")
            except Exception as e:
                self.console.print(f"[yellow]⚠[/yellow] Speech engine failed: {e}")
                self.use_speech = False
        
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
    
    def speak(self, text: str, use_rag: bool = False):
        """
        Make the system speak and log in conversation.
        
        Args:
            text: Text to speak
            use_rag: Whether this response was RAG-based
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.conversation_history.append({
            'time': timestamp,
            'speaker': 'AI',
            'text': text,
            'rag_decision': use_rag
        })
        
        if self.tts_engine and self.use_speech:
            try:
                self.tts_engine.speak(text)
            except Exception as e:
                logger.error(f"TTS error: {e}")
    
    def process_user_input(self, user_text: str):
        """
        Process user input through RAG system and generate response.
        
        Args:
            user_text: User input text
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add user input to conversation
        self.conversation_history.append({
            'time': timestamp,
            'speaker': 'User',
            'text': user_text,
            'rag_decision': False
        })
        
        if not self.use_rag or self.rag_system is None:
            response = f"You said: {user_text}. I understand."
            self.speak(response, use_rag=False)
            return
        
        # Process through RAG system
        try:
            # Create embedding for user input (simplified)
            rng = np.random.default_rng()
            user_embedding = rng.normal(0, 0.1, 128)
            
            # Retrieve context from RAG
            rag_context = self.rag_system.retrieve_context(
                query_embedding=user_embedding,
                top_k=3
            )
            
            # Log RAG decision
            decision_log = {
                'timestamp': timestamp,
                'user_input': user_text,
                'retrieved_count': rag_context.get('retrieved_count', 0),
                'context_relevance': rag_context.get('context_relevance', 0.0),
                'memory_types': rag_context.get('memory_types', [])
            }
            self.rag_decision_log.append(decision_log)
            
            # Generate response based on RAG context
            relevance = rag_context.get('context_relevance', 0.0)
            retrieved = rag_context.get('retrieved_count', 0)
            memory_types = rag_context.get('memory_types', [])
            
            # Build intelligent response using RAG context
            if relevance > 0.6 and retrieved > 0:
                memory_str = ", ".join(memory_types) if memory_types else "multiple modalities"
                response = (f"Based on {retrieved} memories from {memory_str}, "
                           f"I'm {relevance:.0%} confident: That relates to what I've learned!")
            elif relevance > 0.3 and retrieved > 0:
                response = (f"Interesting! I found {retrieved} related memories. "
                           f"This connects to my past with {relevance:.0%} certainty.")
            else:
                response = ("That's new to me! I'm storing this experience in my memory. "
                           "This will help me learn and grow.")
            
            self.speak(response, use_rag=True)
            
            # Update cognitive state with RAG context
            if self.rag_system:
                self.rag_system.update_cognitive_state(rag_context=rag_context)
            
        except Exception as e:
            logger.error(f"Error processing user input through RAG: {e}")
            response = f"I understood: {user_text}. Let me think about that."
            self.speak(response, use_rag=False)
    
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
    
    def create_layout(self, video_ascii: str = "", state_info: dict = None,
                     rag_info: dict = None) -> Layout:
        """
        Create the terminal layout.
        
        Args:
            video_ascii: ASCII art video frame
            state_info: Current system state
            rag_info: RAG system information
            
        Returns:
            Rich Layout object
        """
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="rag_info", size=3),
            Layout(name="footer", size=8)
        )
        
        # Header
        header_text = Text()
        header_text.append("╔═══════════════════════════════════════════════════════════════════╗\n", 
                          style="bold green")
        header_text.append("║  ", style="green")
        header_text.append("COGNITIVE SYSTEM - TERMINAL WITH RAG", style="bold green")
        header_text.append("         ║\n", style="green")
        header_text.append("╚═══════════════════════════════════════════════════════════════════╝", 
                          style="bold green")
        layout["header"].update(Panel(header_text, border_style="green", box=box.HEAVY))
        
        # Main area - video and state
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
        
        # RAG Info panel
        if rag_info and self.use_rag:
            rag_text = (f"[green]RAG Memories: {rag_info.get('retrieved_count', 0)} | "
                       f"Relevance: {rag_info.get('context_relevance', 0.0):.1%}[/green]")
        else:
            rag_text = "[dim green]RAG system inactive[/dim green]"
        
        rag_panel = Panel(
            rag_text,
            title="[bold green]🧠 RAG System[/bold green]",
            border_style="green",
            box=box.ROUNDED
        )
        layout["rag_info"].update(rag_panel)
        
        # Footer - conversation
        conversation_text = ""
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            if msg['speaker'] == 'AI':
                rag_marker = " [RAG]" if msg.get('rag_decision') else ""
                speaker_color = "bold green"
                conversation_text += f"[{speaker_color}]{msg['time']} AI{rag_marker}:[/{speaker_color}] [green]{msg['text']}[/green]\n"
            else:
                conversation_text += f"[green]{msg['time']} User:[/green] [yellow]{msg['text']}[/yellow]\n"
        
        if not conversation_text:
            conversation_text = "[dim green]Chat will appear here...[/dim green]"
        
        footer_panel = Panel(
            conversation_text,
            title="[bold green]💬 Conversation[/bold green]",
            border_style="green",
            box=box.HEAVY
        )
        layout["footer"].update(footer_panel)
        
        return layout
    
    def input_thread(self):
        """Thread for collecting user input."""
        import sys
        while self.running:
            try:
                # Use stdin directly instead of console.input for better compatibility
                sys.stdout.write("\n[You]: ")
                sys.stdout.flush()
                user_input = sys.stdin.readline().strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    self.running = False
                    break
                    
                if user_input.strip():
                    self.user_input_queue.put(user_input)
            except EOFError:
                self.running = False
                break
            except Exception as e:
                logger.debug(f"Input error: {e}")
    
    def run(self):
        """Run the terminal application."""
        self.running = True
        
        # Initial greeting
        greeting = ("Hello! I am your cognitive system. I can see through the camera, "
                   "hear you, chat with you, and I use a special RAG system to remember "
                   "and learn from our interactions. Type 'exit' to quit.")
        self.speak(greeting, use_rag=False)
        
        # Start user input thread
        input_thread = threading.Thread(target=self.input_thread, daemon=True)
        input_thread.start()
        
        try:
            with Live(self.create_layout(), console=self.console, refresh_per_second=4, 
                     screen=True) as live:
                last_speech_time = time.time()
                speech_interval = 5.0
                
                while self.running:
                    try:
                        # Get video frame
                        video_ascii = ""
                        visual_features = None
                        
                        if self.use_camera and self.camera and self.camera.isOpened():
                            ret, frame = self.camera.read()
                            if ret:
                                frame = cv2.flip(frame, 1)
                                video_ascii = self.video_display.frame_to_ascii(frame)
                                visual_features = self.extract_visual_features(frame)
                                
                                # Process through RAG
                                if self.use_rag and self.rag_system:
                                    try:
                                        self.rag_system.process_video_frame(frame, store_embedding=True)
                                    except Exception as e:
                                        logger.debug(f"RAG video processing error: {e}")
                        
                        # Use synthetic features if no camera
                        if visual_features is None:
                            rng = np.random.default_rng()
                            visual_features = rng.normal(0, 0.1, 64)
                        
                        # Synthetic audio features
                        rng = np.random.default_rng()
                        auditory_features = rng.normal(0, 0.05, 32)
                        
                        # Update cognitive system
                        sensory_input = {
                            'visual': visual_features[:10],
                            'auditory': auditory_features[:10],
                        }
                        self.cognitive_system.update(1.0 / 15, sensory_input)
                        
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
                        
                        # Get RAG info if available
                        rag_info = None
                        if self.use_rag and self.rag_system:
                            rag_info = {
                                'retrieved_count': len(self.rag_system.vector_store.store) if hasattr(
                                    self.rag_system.vector_store, 'store') else 0,
                                'context_relevance': 0.0
                            }
                        
                        # Process user input if available
                        if not self.user_input_queue.empty():
                            user_input = self.user_input_queue.get()
                            self.process_user_input(user_input)
                        
                        # Periodic AI responses
                        current_time = time.time()
                        if current_time - last_speech_time > speech_interval:
                            if len(self.conversation_history) == 0 or \
                               self.conversation_history[-1]['speaker'] != 'AI':
                                response = self.generate_response(self.current_state)
                                if response:
                                    self.speak(response, use_rag=False)
                                last_speech_time = current_time
                        
                        # Update display
                        layout = self.create_layout(video_ascii, self.current_state, rag_info)
                        live.update(layout)
                        
                        self.frame_count += 1
                        time.sleep(1.0 / 4)
                        
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
            Response text or empty string
        """
        arousal = state.get('arousal', 0.5)
        mood = state.get('mood', 0.5)
        stress = state.get('stress', 0.0)
        
        responses = []
        
        if arousal > 0.7:
            responses.append("I'm feeling quite alert and focused right now.")
        elif arousal < 0.3:
            responses.append("I'm in a calm and relaxed state.")
        
        if mood > 0.6:
            responses.append("I'm experiencing positive emotions.")
        elif mood < 0.4:
            responses.append("My mood is somewhat neutral at the moment.")
        
        if stress > 0.5:
            responses.append("I'm detecting elevated stress levels in the environment.")
        
        if self.use_camera:
            responses.append("I can see you through the camera.")
        
        if responses:
            import random
            return random.choice(responses)
        return ""
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        self.console.print("\n[green]Shutting down gracefully...[/green]")
        
        if self.cognitive_system:
            self.cognitive_system.stop()
        
        if self.rag_system:
            self.rag_system.stop()
        
        if self.tts_engine:
            self.tts_engine.stop()
        
        if self.camera:
            self.camera.release()
        
        self.console.print("[green]✓ System shutdown complete[/green]\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cognitive System Terminal Interface with RAG"
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
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG system"
    )
    
    args = parser.parse_args()
    
    app = TerminalCognitiveAppWithRAG(
        use_camera=not args.no_camera,
        use_speech=not args.no_speech,
        use_rag=not args.no_rag
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
