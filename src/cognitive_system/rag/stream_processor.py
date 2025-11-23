"""
Stream Processor for Video and Audio

Processes live video and audio streams, extracts features, and generates
embeddings for the RAG system.
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
import logging
import time


class VideoStreamProcessor:
    """
    Processes video streams and extracts visual features.
    
    For a lightweight implementation without external dependencies on computer vision
    libraries, we use simple feature extraction based on statistical properties.
    In a production system, this would use models like CLIP or other vision transformers.
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480,
                 feature_dim: int = 128, temporal_window: int = 5):
        """
        Initialize the video stream processor.
        
        Args:
            frame_width: Width of video frames
            frame_height: Height of video frames
            feature_dim: Dimension of extracted features
            temporal_window: Number of frames to aggregate for temporal features
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.feature_dim = feature_dim
        self.temporal_window = temporal_window
        
        self.frame_buffer = []
        self.frame_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Feature extraction weights (simulated learned features)
        # In production, these would be from a pre-trained vision model
        self.feature_weights = np.random.randn(feature_dim, 64) * 0.1
        
        self.logger.info(f"VideoStreamProcessor initialized: {frame_width}x{frame_height}, feature_dim={feature_dim}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame and extract features.
        
        Args:
            frame: Video frame as numpy array (H, W, C) or (H, W)
            
        Returns:
            Feature vector for the frame
        """
        self.frame_count += 1
        
        # Handle grayscale or RGB
        if len(frame.shape) == 2:
            # Grayscale
            frame_flat = frame.flatten()
        else:
            # RGB - convert to grayscale-like representation
            frame_flat = np.mean(frame, axis=2).flatten()
        
        # Extract statistical features
        # This is a simplified approach; production systems would use CNN/ViT features
        features = self._extract_statistical_features(frame_flat)
        
        # Add to temporal buffer
        self.frame_buffer.append(features)
        if len(self.frame_buffer) > self.temporal_window:
            self.frame_buffer.pop(0)
        
        # Compute temporal features
        temporal_features = self._compute_temporal_features()
        
        return temporal_features
    
    def _extract_statistical_features(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from frame data.
        
        Args:
            frame_data: Flattened frame data
            
        Returns:
            Statistical feature vector
        """
        # Normalize
        if frame_data.max() > frame_data.min():
            frame_data = (frame_data - frame_data.min()) / (frame_data.max() - frame_data.min())
        
        # Compute statistics in spatial patches
        patch_size = max(1, len(frame_data) // 64)
        patches = []
        
        for i in range(0, len(frame_data), patch_size):
            patch = frame_data[i:i+patch_size]
            if len(patch) > 0:
                patches.append([
                    np.mean(patch),
                    np.std(patch),
                    np.max(patch) - np.min(patch)
                ])
        
        # Pad or truncate to 64 patches
        while len(patches) < 64:
            patches.append([0.5, 0.1, 0.3])
        patches = patches[:64]
        
        # Flatten and project to feature dimension
        patch_features = np.array(patches).flatten()[:64]
        features = np.dot(self.feature_weights, patch_features)
        
        # Apply tanh activation
        features = np.tanh(features)
        
        return features
    
    def _compute_temporal_features(self) -> np.ndarray:
        """
        Compute temporal features from frame buffer.
        
        Returns:
            Temporal feature vector
        """
        if len(self.frame_buffer) == 0:
            return np.zeros(self.feature_dim)
        
        # Average recent frames with recency weighting
        weights = np.linspace(0.5, 1.0, len(self.frame_buffer))
        weights = weights / weights.sum()
        
        temporal_features = np.zeros(self.feature_dim)
        for i, features in enumerate(self.frame_buffer):
            temporal_features += weights[i] * features
        
        return temporal_features
    
    def generate_synthetic_frame(self, scene_type: str = 'neutral') -> np.ndarray:
        """
        Generate a synthetic video frame for testing.
        
        Args:
            scene_type: Type of scene ('neutral', 'active', 'calm', 'complex')
            
        Returns:
            Synthetic frame as numpy array
        """
        frame = np.random.randn(self.frame_height, self.frame_width, 3) * 0.3 + 0.5
        
        if scene_type == 'active':
            # Add more high-frequency patterns
            frame += np.random.randn(self.frame_height, self.frame_width, 3) * 0.5
        elif scene_type == 'calm':
            # Smooth patterns - simple averaging as alternative to gaussian filter
            # This avoids scipy dependency while providing similar effect
            kernel_size = 5
            for c in range(3):
                channel = frame[:, :, c]
                smoothed = np.copy(channel)
                for i in range(kernel_size, self.frame_height - kernel_size):
                    for j in range(kernel_size, self.frame_width - kernel_size):
                        smoothed[i, j] = np.mean(
                            channel[i-kernel_size:i+kernel_size+1, 
                                   j-kernel_size:j+kernel_size+1]
                        )
                frame[:, :, c] = smoothed
        elif scene_type == 'complex':
            # Add structured patterns
            x = np.linspace(0, 2 * np.pi, self.frame_width)
            y = np.linspace(0, 2 * np.pi, self.frame_height)
            xx, yy = np.meshgrid(x, y)
            pattern = np.sin(xx) * np.cos(yy)
            frame[:, :, 0] += pattern * 0.3
        
        # Clip to valid range
        frame = np.clip(frame, 0, 1)
        
        return frame
    
    def reset(self):
        """Reset the processor state."""
        self.frame_buffer.clear()
        self.frame_count = 0


class AudioStreamProcessor:
    """
    Processes audio streams and extracts auditory features.
    
    For a lightweight implementation, we use simple spectral and temporal features.
    In production, this would use models like Wav2Vec2 or audio transformers.
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512,
                 feature_dim: int = 128):
        """
        Initialize the audio stream processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Size of audio chunks to process
            feature_dim: Dimension of extracted features
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.feature_dim = feature_dim
        
        self.audio_buffer = []
        self.chunk_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Feature extraction weights
        self.feature_weights = np.random.randn(feature_dim, 32) * 0.1
        
        self.logger.info(f"AudioStreamProcessor initialized: {sample_rate}Hz, chunk_size={chunk_size}, feature_dim={feature_dim}")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process an audio chunk and extract features.
        
        Args:
            audio_chunk: Audio samples as numpy array
            
        Returns:
            Feature vector for the audio chunk
        """
        self.chunk_count += 1
        
        # Ensure correct chunk size
        if len(audio_chunk) < self.chunk_size:
            audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[:self.chunk_size]
        
        # Extract spectral and temporal features
        features = self._extract_audio_features(audio_chunk)
        
        # Add to buffer
        self.audio_buffer.append(features)
        if len(self.audio_buffer) > 10:  # Keep last 10 chunks
            self.audio_buffer.pop(0)
        
        # Compute temporal features
        temporal_features = self._compute_temporal_audio_features()
        
        return temporal_features
    
    def _extract_audio_features(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Extract spectral and temporal features from audio.
        
        Args:
            audio_chunk: Audio samples
            
        Returns:
            Feature vector
        """
        # Simple spectral features using FFT
        # In production, use mel-spectrograms or learned features
        spectrum = np.abs(np.fft.rfft(audio_chunk))
        
        # Divide into frequency bands
        n_bands = 16
        band_size = len(spectrum) // n_bands
        band_features = []
        
        for i in range(n_bands):
            band = spectrum[i*band_size:(i+1)*band_size]
            if len(band) > 0:
                band_features.extend([
                    np.mean(band),
                    np.std(band)
                ])
            else:
                band_features.extend([0.0, 0.0])
        
        # Pad or truncate to 32 features
        band_features = (band_features + [0.0] * 32)[:32]
        
        # Project to feature dimension
        features = np.dot(self.feature_weights, np.array(band_features))
        
        # Apply tanh activation
        features = np.tanh(features)
        
        return features
    
    def _compute_temporal_audio_features(self) -> np.ndarray:
        """
        Compute temporal features from audio buffer.
        
        Returns:
            Temporal feature vector
        """
        if len(self.audio_buffer) == 0:
            return np.zeros(self.feature_dim)
        
        # Average recent chunks with recency weighting
        weights = np.linspace(0.5, 1.0, len(self.audio_buffer))
        weights = weights / weights.sum()
        
        temporal_features = np.zeros(self.feature_dim)
        for i, features in enumerate(self.audio_buffer):
            temporal_features += weights[i] * features
        
        return temporal_features
    
    def generate_synthetic_audio(self, sound_type: str = 'neutral') -> np.ndarray:
        """
        Generate synthetic audio for testing.
        
        Args:
            sound_type: Type of sound ('neutral', 'speech', 'music', 'noise')
            
        Returns:
            Synthetic audio chunk as numpy array
        """
        t = np.linspace(0, self.chunk_size / self.sample_rate, self.chunk_size)
        
        if sound_type == 'speech':
            # Simulate speech-like formants
            audio = 0.3 * np.sin(2 * np.pi * 200 * t)  # F1
            audio += 0.2 * np.sin(2 * np.pi * 600 * t)  # F2
            audio += 0.1 * np.sin(2 * np.pi * 1500 * t)  # F3
            audio += np.random.randn(len(t)) * 0.1  # Noise
        elif sound_type == 'music':
            # Simulate musical tones
            audio = 0.4 * np.sin(2 * np.pi * 440 * t)  # A4
            audio += 0.3 * np.sin(2 * np.pi * 554.37 * t)  # C#5
            audio += 0.2 * np.sin(2 * np.pi * 659.25 * t)  # E5
        elif sound_type == 'noise':
            # White noise
            audio = np.random.randn(len(t)) * 0.5
        else:  # neutral
            # Low-frequency ambient
            audio = 0.2 * np.sin(2 * np.pi * 100 * t)
            audio += np.random.randn(len(t)) * 0.05
        
        return audio
    
    def reset(self):
        """Reset the processor state."""
        self.audio_buffer.clear()
        self.chunk_count = 0
