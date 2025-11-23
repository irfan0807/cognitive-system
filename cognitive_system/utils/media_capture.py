"""Media capture module for live video and audio feeds."""

import cv2
import numpy as np
import threading
import queue
from typing import Optional, Dict, Any
import logging


logger = logging.getLogger(__name__)


class MediaStreamManager:
    """Manages multiple media streams (video, audio)."""
    
    def __init__(self):
        """Initialize the media stream manager."""
        self.streams: Dict[str, Any] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.queues: Dict[str, queue.Queue] = {}
        self.running = False
        
    def add_camera_stream(self, name: str = "camera", camera_id: int = 0, fps: int = 30):
        """Add a camera stream."""
        try:
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.streams[name] = {
                'type': 'camera',
                'capture': cap,
                'fps': fps
            }
            self.queues[name] = queue.Queue(maxsize=2)
            logger.info(f"Camera stream '{name}' added (ID: {camera_id})")
        except Exception as e:
            logger.error(f"Failed to add camera stream: {e}")
            raise
    
    def add_microphone_stream(self, name: str = "microphone", sample_rate: int = 16000, 
                            chunk_size: int = 1024):
        """Add a microphone stream (placeholder)."""
        try:
            import pyaudio
            self.streams[name] = {
                'type': 'microphone',
                'sample_rate': sample_rate,
                'chunk_size': chunk_size
            }
            self.queues[name] = queue.Queue(maxsize=2)
            logger.info(f"Microphone stream '{name}' configured")
        except ImportError:
            logger.warning("PyAudio not available, microphone stream disabled")
        except Exception as e:
            logger.error(f"Failed to add microphone stream: {e}")
            raise
    
    def _capture_camera(self, name: str):
        """Capture frames from camera in a separate thread."""
        stream = self.streams[name]
        cap = stream['capture']
        q = self.queues[name]
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                logger.warning(f"Failed to read frame from camera '{name}'")
                break
        
        cap.release()
    
    def start_all(self):
        """Start all configured streams."""
        self.running = True
        
        for name, stream in self.streams.items():
            if stream['type'] == 'camera':
                thread = threading.Thread(target=self._capture_camera, args=(name,), daemon=True)
                thread.start()
                self.threads[name] = thread
                logger.info(f"Started camera stream '{name}'")
    
    def stop_all(self):
        """Stop all streams."""
        self.running = False
        for thread in self.threads.values():
            thread.join(timeout=1.0)
        logger.info("All streams stopped")
    
    def get_frame(self, name: str = "camera") -> Optional[np.ndarray]:
        """Get the latest frame from a stream."""
        if name not in self.queues:
            return None
        
        try:
            return self.queues[name].get_nowait()
        except queue.Empty:
            return None


class VideoProcessor:
    """Process video frames for neural network input."""
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        """Preprocess a video frame for neural network input."""
        # Resize
        resized = cv2.resize(frame, target_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    @staticmethod
    def extract_features(frame: np.ndarray) -> np.ndarray:
        """Extract simple features from a frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Flatten for feature vector
        features = edges.flatten() / 255.0
        
        # Sample to get reasonable size
        if len(features) > 256:
            indices = np.linspace(0, len(features) - 1, 256, dtype=int)
            features = features[indices]
        
        return features
