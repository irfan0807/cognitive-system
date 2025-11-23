"""
Text-to-Speech Engine for Cognitive System

Provides speech synthesis capabilities so the system can speak to the user.
"""

import logging
import threading
import queue
from typing import Optional

logger = logging.getLogger(__name__)


class TextToSpeechEngine:
    """
    Text-to-Speech engine with thread-safe queuing.
    """
    
    def __init__(self, rate: int = 150, volume: float = 0.9):
        """
        Initialize TTS engine.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.rate = rate
        self.volume = volume
        self.engine = None
        self.speech_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.warning(f"TTS engine initialization failed: {e}")
            self.engine = None
    
    def start(self):
        """Start the TTS worker thread."""
        if not self.engine:
            logger.warning("TTS engine not available, speech disabled")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("TTS worker thread started")
    
    def stop(self):
        """Stop the TTS worker thread."""
        self.running = False
        if self.worker_thread:
            self.speech_queue.put(None)  # Sentinel to exit
            self.worker_thread.join(timeout=2.0)
        logger.info("TTS worker thread stopped")
    
    def _worker(self):
        """Worker thread that processes speech requests."""
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.1)
                if text is None:  # Sentinel value
                    break
                
                if self.engine:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def speak(self, text: str):
        """
        Queue text to be spoken.
        
        Args:
            text: Text to speak
        """
        if not self.engine:
            logger.debug(f"TTS not available, would say: {text}")
            return
        
        self.speech_queue.put(text)
    
    def speak_sync(self, text: str):
        """
        Speak text synchronously (blocks until complete).
        
        Args:
            text: Text to speak
        """
        if not self.engine:
            logger.debug(f"TTS not available, would say: {text}")
            return
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS sync error: {e}")
