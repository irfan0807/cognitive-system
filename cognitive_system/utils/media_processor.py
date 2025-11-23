"""Media input bridge for cognitive system."""

import numpy as np
from typing import Dict, Optional, Any
import logging


logger = logging.getLogger(__name__)


class MediaInputBridge:
    """Bridge between media streams and cognitive system."""
    
    def __init__(self):
        """Initialize the media input bridge."""
        self.last_visual_input = None
        self.last_audio_input = None
    
    def process_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process video frame into sensory input."""
        # Simple feature extraction
        if frame is None:
            return np.zeros(10)
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = 0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0]
        else:
            gray = frame
        
        # Downsample
        downsampled = gray[::4, ::4]
        
        # Extract statistics as features
        features = np.array([
            np.mean(downsampled),
            np.std(downsampled),
            np.min(downsampled),
            np.max(downsampled),
            np.median(downsampled),
            np.percentile(downsampled, 25),
            np.percentile(downsampled, 75),
            np.var(downsampled),
            np.sum(downsampled) / downsampled.size,
            np.prod(downsampled.shape) / 10000.0
        ], dtype=np.float32)
        
        # Normalize to roughly 0-1 range
        features = np.clip(features / 255.0, 0, 1)
        
        self.last_visual_input = features
        return features
    
    def get_sensory_input(self, visual_frame: Optional[np.ndarray] = None,
                         audio_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Get combined sensory input from multiple sources."""
        sensory_input = {}
        
        # Visual input
        if visual_frame is not None:
            sensory_input['visual'] = self.process_video_frame(visual_frame)
        else:
            if self.last_visual_input is not None:
                sensory_input['visual'] = self.last_visual_input
            else:
                sensory_input['visual'] = np.random.randn(10) * 0.1
        
        # Auditory input (placeholder)
        if audio_data is not None:
            sensory_input['auditory'] = np.array(audio_data[:10], dtype=np.float32)
        else:
            sensory_input['auditory'] = np.random.randn(10) * 0.1
        
        return sensory_input
