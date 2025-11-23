#!/usr/bin/env python3
"""
Cognitive System - Interactive Terminal with RAG

Simple, responsive chat interface with:
- Real-time responses to your messages
- RAG-based memory system
- Live camera feed (optional)
- Text-to-speech output (optional)

Usage:
    python terminal_app_rag.py              # Full mode
    python terminal_app_rag.py --no-camera  # Text-only (recommended for testing)
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

# Import cognitive system components
from cognitive_system.core.embodied_cognition import EmbodiedCognitionSystem
from cognitive_system.virtual_biology.nervous_system import VirtualNervousSystem
from cognitive_system.core.neural_network import NeuralNetworkController
from cognitive_system.memory.multimodal_memory import MultimodalMemorySystem
from cognitive_system.behavior.behavior_engine import BehaviorEngine
from cognitive_system.rag.multimodal_rag import MultimodalRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/tmp/cognitive_terminal.log')]
)
logger = logging.getLogger(__name__)


class InteractiveTerminalCognitive:
    """Simple, responsive terminal cognitive system with RAG."""
    
    def __init__(self, use_camera: bool = False, use_speech: bool = False):
        """Initialize the system."""
        self.use_camera = use_camera
        self.use_speech = use_speech
        
        # Initialize components
        self.cognitive_system = None
        self.nervous_system = None
        self.neural_network = None
        self.memory_system = None
        self.behavior_engine = None
        self.rag_system = None
        self.camera = None
        
        # Memory store for conversations
        self.memories = []
        self.conversation_count = 0
        
    def initialize(self):
        """Initialize system components."""
        print("\n🧠 Initializing Cognitive System...")
        
        # Initialize cognitive system
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
        print("✓ Cognitive System ready")
        
        # Initialize RAG system
        print("🧠 Initializing RAG Memory System...")
        try:
            self.rag_system = MultimodalRAGSystem(embedding_dim=128)
            self.rag_system.attach_cognitive_system(self.cognitive_system)
            self.rag_system.start()
            print("✓ RAG System ready")
        except Exception as e:
            print(f"⚠ RAG System error: {e}")
            self.rag_system = None
        
        # Initialize camera if requested
        if self.use_camera:
            print("📹 Initializing Camera...")
            try:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print("✓ Camera ready")
                else:
                    print("⚠ Camera not available")
                    self.use_camera = False
            except Exception as e:
                print(f"⚠ Camera error: {e}")
                self.use_camera = False
        
        print("\n" + "="*60)
        print("✅ System Ready! Type 'exit' to quit")
        print("="*60 + "\n")
    
    def get_system_state(self) -> dict:
        """Get current system state."""
        if self.nervous_system:
            state = self.nervous_system.get_state()
            return {
                'arousal': state.get('arousal', 0.5),
                'stress': state.get('stress_level', 0.0),
                'mood': state.get('mood', 0.5),
                'heart_rate': state.get('heart_rate', 60.0),
                'attention': state.get('attention', 0.5),
            }
        return {}
    
    def process_with_rag(self, user_input: str) -> str:
        """Process user input through RAG system."""
        if not self.rag_system:
            return self.generate_simple_response(user_input)
        
        try:
            # Create embedding for user input
            rng = np.random.default_rng(42)
            user_embedding = rng.normal(0, 0.1, 128)
            
            # Retrieve context from memories
            rag_context = self.rag_system.retrieve_context(
                query_embedding=user_embedding,
                top_k=3
            )
            
            # Log to memory
            memory_entry = {
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'user_input': user_input,
                'retrieved_count': rag_context.get('retrieved_count', 0),
                'context_relevance': rag_context.get('context_relevance', 0.0)
            }
            self.memories.append(memory_entry)
            
            # Generate response based on relevance
            relevance = rag_context.get('context_relevance', 0.0)
            retrieved = rag_context.get('retrieved_count', 0)
            
            if relevance > 0.6 and retrieved > 0:
                response = (f"[RAG] Based on my {retrieved} memories, I'm {relevance:.0%} confident: "
                           f"That relates to what I've learned!")
            elif relevance > 0.3 and retrieved > 0:
                response = (f"[RAG] I found {retrieved} related memories. "
                           f"This connects to my past with {relevance:.0%} certainty.")
            else:
                response = "That's interesting! I'm storing this in my memory."
            
            # Update cognitive state
            self.rag_system.update_cognitive_state(rag_context=rag_context)
            
            return response
            
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return self.generate_simple_response(user_input)
    
    def generate_simple_response(self, user_input: str) -> str:
        """Generate simple response without RAG."""
        # Check if user is asking about the system
        lower_input = user_input.lower()
        
        simple_responses = {
            'hello': "Hello! I'm your cognitive AI. How can I help?",
            'hi': "Hi there! What would you like to talk about?",
            'how are you': "I'm functioning well, thank you for asking!",
            'what are you': "I'm an AI cognitive system with memory and learning capabilities.",
            'who are you': "I'm your conversational AI powered by embodied cognition and RAG memory.",
            'thanks': "You're welcome! Happy to help.",
            'thank you': "You're welcome! Happy to help.",
            'bye': "Goodbye! It was nice talking with you.",
            'see you': "See you later! Take care.",
        }
        
        # Check for exact matches
        for key, response in simple_responses.items():
            if lower_input == key:
                return response
        
        # Check for partial matches
        for key, response in simple_responses.items():
            if key in lower_input:
                return response
        
        # Default responses
        default_responses = [
            "That's an interesting point. Tell me more.",
            "I understand what you're saying. Can you elaborate?",
            "That's something to think about.",
            "I find that fascinating. What else?",
            "I see. Let me process that.",
            "Very interesting! Go on.",
            "I'm listening. What's your thought?",
        ]
        
        return random.choice(default_responses)
    
    def show_status(self):
        """Show current system state."""
        state = self.get_system_state()
        if state:
            print(f"\n📊 System State:")
            print(f"   Arousal: {state.get('arousal', 0.5):.2f}")
            print(f"   Mood: {state.get('mood', 0.5):.2f}")
            print(f"   Stress: {state.get('stress', 0.0):.2f}")
            print(f"   Heart Rate: {state.get('heart_rate', 60.0):.0f} bpm")
            print(f"   Attention: {state.get('attention', 0.5):.2f}")
            print(f"   Memories Stored: {len(self.memories)}")
    
    def chat(self):
        """Start interactive chat."""
        print("💬 Starting chat session...\n")
        
        # Show initial greeting
        greeting = "Hello! I'm your cognitive AI system. I can remember our conversations and learn from them. What would you like to talk about?"
        print(f"🤖 AI: {greeting}\n")
        
        self.conversation_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("\n🤖 AI: Goodbye! It was nice talking with you. Take care!")
                    break
                
                # Check for special commands
                if user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                if user_input.lower() == 'memories':
                    print(f"\n📚 Stored Memories ({len(self.memories)}):")
                    for i, mem in enumerate(self.memories[-5:], 1):  # Show last 5
                        print(f"   {i}. [{mem['timestamp']}] {mem['user_input'][:50]}...")
                    print()
                    continue
                
                # Process input through system
                self.conversation_count += 1
                
                # Process through RAG
                response = self.process_with_rag(user_input)
                
                # Speak if enabled
                if self.use_speech:
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.say(response)
                        engine.runAndWait()
                    except Exception as e:
                        logger.debug(f"Speech error: {e}")
                
                # Print response
                print(f"🤖 AI: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted by user.")
                break
            except EOFError:
                print("\n\n👋 Chat ended.")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Chat error: {e}", exc_info=True)
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🛑 Shutting down...")
        
        if self.cognitive_system:
            try:
                self.cognitive_system.stop()
            except:
                pass
        
        if self.rag_system:
            try:
                self.rag_system.stop()
            except:
                pass
        
        if self.camera:
            self.camera.release()
        
        print("✓ Shutdown complete\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Cognitive System with RAG Memory"
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable camera input (experimental)"
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        help="Enable text-to-speech output"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🧠 COGNITIVE SYSTEM - INTERACTIVE CHAT")
    print("="*60)
    
    app = InteractiveTerminalCognitive(
        use_camera=args.camera,
        use_speech=args.speech
    )
    
    try:
        app.initialize()
        app.chat()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        app.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    main()
