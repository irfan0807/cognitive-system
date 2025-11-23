#!/usr/bin/env python3
"""
Cognitive System - Interactive Chat with Local LLM & RAG

Powerful conversational AI combining:
- Local LLM for intelligent responses
- RAG system for memory and context
- Cognitive system with embodied cognition
- Real-time learning and adaptation

Usage:
    python interactive_llm_rag.py              # Full mode
    python interactive_llm_rag.py --no-llm     # Without LLM
    python interactive_llm_rag.py --speech     # With speech
"""

import sys
import os
import time
import logging
import argparse
import random
from datetime import datetime
from typing import Optional, Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    handlers=[logging.FileHandler('/tmp/cognitive_llm.log')]
)
logger = logging.getLogger(__name__)


class LocalLLMManager:
    """Manage local LLM integration."""
    
    def __init__(self, use_llm: bool = True):
        """Initialize LLM manager."""
        self.use_llm = use_llm
        self.llm_model = None
        self.llm_available = False
        
        if use_llm:
            self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize local LLM."""
        try:
            # Try to use ollama (most popular local LLM)
            try:
                import ollama
                self.llm_model = "ollama"
                self.llm_available = True
                logger.info("Ollama LLM available")
                return
            except ImportError:
                pass
            
            # Try to use llama-cpp-python
            try:
                from llama_cpp import Llama
                self.llm_model = "llama_cpp"
                self.llm_available = True
                logger.info("Llama.cpp LLM available")
                return
            except ImportError:
                pass
            
            # Try to use transformers with local model
            try:
                from transformers import pipeline
                self.llm_model = "transformers"
                self.llm_available = True
                logger.info("Transformers LLM available")
                return
            except ImportError:
                pass
            
            # Try to use GPT4All
            try:
                from gpt4all import GPT4All
                self.llm_model = "gpt4all"
                self.llm_available = True
                logger.info("GPT4All LLM available")
                return
            except ImportError:
                pass
            
            logger.warning("No local LLM library found. Using fallback responses.")
            self.llm_available = False
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm_available = False
    
    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 100) -> str:
        """Generate response using local LLM."""
        if not self.llm_available:
            return None
        
        try:
            full_prompt = f"{context}\n\nUser: {prompt}\n\nAI:"
            
            if self.llm_model == "ollama":
                try:
                    import ollama
                    response = ollama.generate(
                        model="neural-chat",  # or any other model you have
                        prompt=full_prompt,
                        stream=False,
                        options={
                            "num_predict": max_tokens,
                            "temperature": 0.7,
                        }
                    )
                    return response.get('response', '').strip()
                except Exception as e:
                    logger.debug(f"Ollama error: {e}")
                    return None
            
            elif self.llm_model == "llama_cpp":
                try:
                    from llama_cpp import Llama
                    llm = Llama(model_path="/path/to/model.gguf")
                    output = llm(
                        full_prompt,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        stop=["User:", "AI:"]
                    )
                    return output['choices'][0]['text'].strip()
                except Exception as e:
                    logger.debug(f"Llama.cpp error: {e}")
                    return None
            
            elif self.llm_model == "transformers":
                try:
                    from transformers import pipeline
                    generator = pipeline("text-generation", model="distilgpt2")
                    output = generator(full_prompt, max_length=150, num_return_sequences=1)
                    return output[0]['generated_text'].replace(full_prompt, '').strip()
                except Exception as e:
                    logger.debug(f"Transformers error: {e}")
                    return None
            
            elif self.llm_model == "gpt4all":
                try:
                    from gpt4all import GPT4All
                    model = GPT4All("orca-mini-3b")
                    response = model.generate(
                        full_prompt,
                        max_tokens=max_tokens,
                        temp=0.7
                    )
                    return response.strip()
                except Exception as e:
                    logger.debug(f"GPT4All error: {e}")
                    return None
        
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None


class PowerfulCognitiveChat:
    """Powerful cognitive system with LLM + RAG."""
    
    def __init__(self, use_llm: bool = True, use_speech: bool = False):
        """Initialize the system."""
        self.use_llm = use_llm
        self.use_speech = use_speech
        
        # Initialize components
        self.cognitive_system = None
        self.nervous_system = None
        self.neural_network = None
        self.memory_system = None
        self.behavior_engine = None
        self.rag_system = None
        self.llm_manager = None
        
        # Memory and context
        self.memories: List[Dict] = []
        self.conversation_history: List[Dict] = []
        self.conversation_count = 0
    
    def initialize(self):
        """Initialize all system components."""
        print("\n" + "="*60)
        print("🧠 POWERFUL COGNITIVE SYSTEM WITH LOCAL LLM")
        print("="*60)
        
        print("\n🧠 Initializing Cognitive System...")
        try:
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
        except Exception as e:
            print(f"⚠ Cognitive System error: {e}")
            logger.error(f"Cognitive system init error: {e}")
        
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
        
        # Initialize LLM
        if self.use_llm:
            print("🤖 Initializing Local LLM...")
            self.llm_manager = LocalLLMManager(use_llm=True)
            if self.llm_manager.llm_available:
                print(f"✓ LLM ready ({self.llm_manager.llm_model})")
            else:
                print("⚠ LLM not available, using fallback responses")
        
        print("\n" + "="*60)
        print("✅ System Ready!")
        print("="*60)
        print("\nTry these special commands:")
        print("  'status'   - Show system state")
        print("  'memories' - Show stored memories")
        print("  'context'  - Show conversation context")
        print("  'exit'     - Quit\n")
    
    def get_system_state(self) -> Dict:
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
    
    def build_context(self, user_input: str) -> str:
        """Build context for LLM from RAG memories."""
        context_parts = []
        
        # Add system state
        state = self.get_system_state()
        if state:
            context_parts.append(
                f"[Current State] Arousal: {state.get('arousal', 0.5):.2f}, "
                f"Mood: {state.get('mood', 0.5):.2f}, "
                f"Stress: {state.get('stress', 0.0):.2f}"
            )
        
        # Add relevant memories from RAG
        if self.rag_system:
            try:
                rng = np.random.default_rng(42)
                user_embedding = rng.normal(0, 0.1, 128)
                rag_context = self.rag_system.retrieve_context(
                    query_embedding=user_embedding,
                    top_k=3
                )
                
                retrieved = rag_context.get('retrieved_count', 0)
                if retrieved > 0:
                    context_parts.append(
                        f"[Relevant Memories] Found {retrieved} related past experiences"
                    )
            except Exception as e:
                logger.debug(f"RAG context error: {e}")
        
        # Add recent conversation history
        if len(self.conversation_history) > 0:
            recent = self.conversation_history[-3:]
            history_text = "\n".join([
                f"{msg['speaker']}: {msg['text'][:100]}"
                for msg in recent
            ])
            context_parts.append(f"[Recent Chat]\n{history_text}")
        
        return "\n".join(context_parts)
    
    def generate_powerful_response(self, user_input: str) -> tuple[str, bool]:
        """Generate response using LLM + RAG."""
        is_llm_response = False
        
        # Try LLM first if available
        if self.use_llm and self.llm_manager and self.llm_manager.llm_available:
            try:
                context = self.build_context(user_input)
                llm_response = self.llm_manager.generate_response(
                    prompt=user_input,
                    context=context,
                    max_tokens=150
                )
                
                if llm_response and len(llm_response) > 10:
                    is_llm_response = True
                    return llm_response, is_llm_response
            except Exception as e:
                logger.debug(f"LLM generation error: {e}")
        
        # Fallback to RAG-based response
        if self.rag_system:
            try:
                rng = np.random.default_rng(42)
                user_embedding = rng.normal(0, 0.1, 128)
                rag_context = self.rag_system.retrieve_context(
                    query_embedding=user_embedding,
                    top_k=3
                )
                
                retrieved = rag_context.get('retrieved_count', 0)
                relevance = rag_context.get('context_relevance', 0.0)
                
                if relevance > 0.6 and retrieved > 0:
                    response = (f"[RAG] Based on my {retrieved} memories, I'm {relevance:.0%} "
                               f"confident: That relates to what I've learned!")
                elif relevance > 0.3 and retrieved > 0:
                    response = (f"[RAG] I found {retrieved} related memories. "
                               f"This connects to my past with {relevance:.0%} certainty.")
                else:
                    response = "That's interesting! I'm storing this in my memory."
                
                return response, is_llm_response
            except Exception as e:
                logger.debug(f"RAG error: {e}")
        
        # Final fallback
        return "I understand. Tell me more.", is_llm_response
    
    def show_status(self):
        """Show system status."""
        state = self.get_system_state()
        print(f"\n📊 System Status:")
        print(f"   Arousal:     {state.get('arousal', 0.5):.2f}")
        print(f"   Mood:        {state.get('mood', 0.5):.2f}")
        print(f"   Stress:      {state.get('stress', 0.0):.2f}")
        print(f"   Heart Rate:  {state.get('heart_rate', 60.0):.0f} bpm")
        print(f"   Attention:   {state.get('attention', 0.5):.2f}")
        print(f"   Conversations: {self.conversation_count}")
        print(f"   Memories:    {len(self.memories)}")
        if self.llm_manager:
            status = "✓ Active" if self.llm_manager.llm_available else "✗ Not available"
            print(f"   LLM:         {status}")
    
    def chat(self):
        """Start interactive chat."""
        print("💬 Starting chat session...\n")
        
        greeting = (
            "Hello! I'm a powerful cognitive AI powered by a local language model and "
            "retrieval-augmented memory. I can have intelligent conversations and learn "
            "from our interactions. What would you like to talk about?"
        )
        print(f"🤖 AI: {greeting}\n")
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for special commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("\n🤖 AI: Goodbye! It was wonderful talking with you. Take care!")
                    break
                
                if user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                if user_input.lower() == 'memories':
                    print(f"\n📚 Stored Memories ({len(self.memories)}):")
                    for i, mem in enumerate(self.memories[-5:], 1):
                        print(f"   {i}. [{mem['timestamp']}] {mem['user_input'][:60]}...")
                    print()
                    continue
                
                if user_input.lower() == 'context':
                    context = self.build_context(user_input)
                    print(f"\n📖 Current Context:\n{context}\n")
                    continue
                
                # Process input
                self.conversation_count += 1
                self.conversation_history.append({
                    'speaker': 'User',
                    'text': user_input,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Generate response
                response, is_llm = self.generate_powerful_response(user_input)
                
                # Log memory
                self.memories.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'user_input': user_input,
                    'response_type': 'LLM' if is_llm else 'RAG'
                })
                
                # Add to conversation history
                self.conversation_history.append({
                    'speaker': 'AI',
                    'text': response,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'is_llm': is_llm
                })
                
                # Speak if enabled
                if self.use_speech:
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.say(response)
                        engine.runAndWait()
                    except Exception as e:
                        logger.debug(f"Speech error: {e}")
                
                # Print response with marker
                marker = "🤖 AI [LLM]:" if is_llm else "🤖 AI:"
                print(f"{marker} {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted.")
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
            except Exception:
                pass
        
        if self.rag_system:
            try:
                self.rag_system.stop()
            except Exception:
                pass
        
        print("✓ Shutdown complete\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Powerful Cognitive AI with Local LLM + RAG Memory"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable local LLM (use RAG only)"
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        help="Enable text-to-speech output"
    )
    
    args = parser.parse_args()
    
    app = PowerfulCognitiveChat(
        use_llm=not args.no_llm,
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
