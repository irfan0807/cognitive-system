"""
Live Feed Processing for Video and Audio

Captures and processes live video and audio feeds for the consciousness system.
"""

import numpy as np
import threading
import queue
import logging
from typing import Optional, Callable, Tuple, Dict, Any
import time

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class LiveVideoFeed:
    """
    Captures and processes live video feed from camera.
    
    Provides preprocessed video frames for the deep neural network.
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 target_size: Tuple[int, int] = (224, 224),
                 fps: int = 30):
        """
        Initialize live video feed.
        
        Args:
            camera_id: Camera device ID (0 for default camera)
            target_size: Target size for video frames (width, height)
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.target_size = target_size
        self.fps = fps
        self.capture = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread = None
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> bool:
        """
        Start capturing video feed.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not CV2_AVAILABLE:
            self.logger.warning("OpenCV not available, cannot start video feed")
            return False
            
        try:
            self.capture = cv2.VideoCapture(self.camera_id)
            if not self.capture.isOpened():
                self.logger.error(f"Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_size[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.logger.info(f"Video feed started: camera {self.camera_id}, "
                           f"size {self.target_size}, fps {self.fps}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting video feed: {e}")
            return False
    
    def stop(self):
        """Stop capturing video feed."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()
        self.logger.info("Video feed stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.is_running:
            ret, frame = self.capture.read()
            if not ret:
                self.logger.warning("Failed to read frame")
                continue
            
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Add to queue (drop oldest if full)
            try:
                self.frame_queue.put_nowait(processed_frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # Remove oldest
                    self.frame_queue.put_nowait(processed_frame)
                except queue.Empty:
                    pass
            
            # Control frame rate
            time.sleep(1.0 / self.fps)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess video frame.
        
        Args:
            frame: Raw frame from camera
            
        Returns:
            Preprocessed frame
        """
        if not CV2_AVAILABLE:
            return frame
            
        # Resize if needed
        if frame.shape[:2] != self.target_size[::-1]:
            frame = cv2.resize(frame, self.target_size)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get latest video frame.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Preprocessed frame or None if not available
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_simulated_frame(self) -> np.ndarray:
        """
        Get simulated frame for testing without camera.
        
        Returns:
            Simulated video frame
        """
        # Generate synthetic frame with some structure
        frame = np.random.rand(*self.target_size[::-1], 3).astype(np.float32) * 0.3
        
        # Add some geometric patterns
        center = (self.target_size[0] // 2, self.target_size[1] // 2)
        radius = min(self.target_size) // 4
        y, x = np.ogrid[:self.target_size[1], :self.target_size[0]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        frame[mask] += 0.5
        
        return np.clip(frame, 0, 1).astype(np.float32)


class LiveAudioFeed:
    """
    Captures and processes live audio feed from microphone.
    
    Provides audio features for the deep neural network.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 chunk_duration: float = 1.0,
                 n_mels: int = 128,
                 n_fft: int = 2048):
        """
        Initialize live audio feed.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of audio chunks in seconds
            n_mels: Number of mel frequency bands
            n_fft: FFT window size
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        
        self.is_running = False
        self.audio_buffer = []
        self.feature_queue = queue.Queue(maxsize=10)
        self.stream = None
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> bool:
        """
        Start capturing audio feed.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not SOUNDDEVICE_AVAILABLE:
            self.logger.warning("sounddevice not available, cannot start audio feed")
            return False
            
        try:
            self.is_running = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=self.chunk_size // 4
            )
            self.stream.start()
            
            self.logger.info(f"Audio feed started: sample_rate {self.sample_rate}, "
                           f"chunk_duration {self.chunk_duration}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting audio feed: {e}")
            return False
    
    def stop(self):
        """Stop capturing audio feed."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.logger.info("Audio feed stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input."""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Add to buffer
        self.audio_buffer.extend(indata[:, 0].tolist())
        
        # Process when we have enough data
        if len(self.audio_buffer) >= self.chunk_size:
            chunk = np.array(self.audio_buffer[:self.chunk_size])
            self.audio_buffer = self.audio_buffer[self.chunk_size // 2:]  # 50% overlap
            
            # Extract features
            features = self._extract_features(chunk)
            
            # Add to queue
            try:
                self.feature_queue.put_nowait(features)
            except queue.Full:
                try:
                    self.feature_queue.get_nowait()
                    self.feature_queue.put_nowait(features)
                except queue.Empty:
                    pass
    
    def _extract_features(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Extract audio features (mel spectrogram).
        
        Args:
            audio_chunk: Raw audio data
            
        Returns:
            Mel spectrogram features
        """
        if not LIBROSA_AVAILABLE:
            # Return simulated features if librosa not available
            time_steps = int(self.chunk_duration * 100)
            return np.random.randn(time_steps, self.n_mels).astype(np.float32) * 0.5
            
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
        
        # Transpose to (time, frequency)
        return log_mel_spec.T
    
    def get_features(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get latest audio features.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio features or None if not available
        """
        try:
            return self.feature_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_simulated_features(self) -> np.ndarray:
        """
        Get simulated audio features for testing without microphone.
        
        Returns:
            Simulated audio features
        """
        # Generate synthetic mel spectrogram
        # Simulate some temporal structure
        time_steps = int(self.chunk_duration * 100)  # 100 Hz temporal resolution
        features = np.random.randn(time_steps, self.n_mels) * 0.5
        
        # Add some harmonic structure
        for i in range(5):
            freq_idx = np.random.randint(0, self.n_mels)
            features[:, freq_idx] += np.sin(np.linspace(0, 4 * np.pi, time_steps))
        
        return features.astype(np.float32)


class LiveFeedManager:
    """
    Manages both video and audio live feeds.
    
    Coordinates capture and provides synchronized multimodal input.
    """
    
    def __init__(self,
                 use_camera: bool = True,
                 use_microphone: bool = True,
                 camera_id: int = 0,
                 video_size: Tuple[int, int] = (224, 224),
                 sample_rate: int = 16000):
        """
        Initialize live feed manager.
        
        Args:
            use_camera: Whether to use real camera
            use_microphone: Whether to use real microphone
            camera_id: Camera device ID
            video_size: Target video frame size
            sample_rate: Audio sample rate
        """
        self.use_camera = use_camera
        self.use_microphone = use_microphone
        
        self.video_feed = LiveVideoFeed(camera_id=camera_id, target_size=video_size)
        self.audio_feed = LiveAudioFeed(sample_rate=sample_rate)
        
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> bool:
        """
        Start all feeds.
        
        Returns:
            True if started successfully
        """
        success = True
        
        if self.use_camera:
            if not self.video_feed.start():
                self.logger.warning("Camera not available, will use simulated video")
                self.use_camera = False
        
        if self.use_microphone:
            if not self.audio_feed.start():
                self.logger.warning("Microphone not available, will use simulated audio")
                self.use_microphone = False
        
        if not self.use_camera and not self.use_microphone:
            self.logger.info("Using simulated feeds for both video and audio")
        
        return success
    
    def stop(self):
        """Stop all feeds."""
        if self.use_camera:
            self.video_feed.stop()
        if self.use_microphone:
            self.audio_feed.stop()
    
    def get_multimodal_input(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get synchronized video and audio input.
        
        Returns:
            Tuple of (video_frame, audio_features)
        """
        # Get video frame
        if self.use_camera:
            video_frame = self.video_feed.get_frame(timeout=0.1)
            if video_frame is None:
                video_frame = self.video_feed.get_simulated_frame()
        else:
            video_frame = self.video_feed.get_simulated_frame()
        
        # Get audio features
        if self.use_microphone:
            audio_features = self.audio_feed.get_features(timeout=0.1)
            if audio_features is None:
                audio_features = self.audio_feed.get_simulated_features()
        else:
            audio_features = self.audio_feed.get_simulated_features()
        
        return video_frame, audio_features
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
