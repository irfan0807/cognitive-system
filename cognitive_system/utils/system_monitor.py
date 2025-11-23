"""
System Monitor for Cognitive System

Provides real-time monitoring and debugging of:
- Video feed processing
- RAG system interactions
- Neural network responses
- Memory consolidation
- Physiological states
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FrameMetrics:
    """Metrics for a single frame processing."""
    frame_number: int
    timestamp: float
    video_received: bool
    video_shape: Optional[tuple] = None
    video_mean_intensity: float = 0.0
    
    visual_features_extracted: bool = False
    visual_feature_dim: int = 0
    visual_feature_mean: float = 0.0
    visual_feature_std: float = 0.0
    
    rag_retrieved: bool = False
    rag_retrieved_count: int = 0
    rag_context_relevance: float = 0.0
    rag_memory_types: List[str] = field(default_factory=list)
    
    neural_network_processed: bool = False
    neural_network_output_dim: int = 0
    neural_network_output_mean: float = 0.0
    
    animation_updated: bool = False
    animation_state: Optional[Dict[str, float]] = None
    
    processing_time_ms: float = 0.0
    
    def to_string(self) -> str:
        """Convert to readable string."""
        parts = []
        parts.append(f"Frame {self.frame_number:4d} @ {datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S.%f')[:-3]}")
        
        if self.video_received:
            parts.append(f"✓ Video: {self.video_shape} (intensity: {self.video_mean_intensity:.3f})")
        else:
            parts.append("✗ Video: Not received")
        
        if self.visual_features_extracted:
            parts.append(f"✓ Features: {self.visual_feature_dim}D (μ={self.visual_feature_mean:.3f}, σ={self.visual_feature_std:.3f})")
        else:
            parts.append("✗ Features: Not extracted")
        
        if self.rag_retrieved:
            parts.append(f"✓ RAG: Retrieved {self.rag_retrieved_count} ({self.rag_context_relevance:.2f} relevance) [{', '.join(self.rag_memory_types)}]")
        else:
            parts.append("✗ RAG: No retrieval")
        
        if self.neural_network_processed:
            parts.append(f"✓ NN: {self.neural_network_output_dim}D (μ={self.neural_network_output_mean:.3f})")
        else:
            parts.append("✗ NN: Not processed")
        
        if self.animation_updated and self.animation_state:
            state_str = ", ".join([f"{k}={v:.2f}" for k, v in list(self.animation_state.items())[:3]])
            parts.append(f"✓ Animation: [{state_str}]")
        else:
            parts.append("✗ Animation: Not updated")
        
        parts.append(f"| {self.processing_time_ms:.2f}ms")
        
        return " | ".join(parts)


class SystemMonitor:
    """Monitor and debug the cognitive system."""
    
    def __init__(self, history_size: int = 100, verbose: bool = True):
        """
        Initialize system monitor.
        
        Args:
            history_size: Number of frames to keep in history
            verbose: Enable verbose logging
        """
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.history_size = history_size
        self.frame_metrics: deque = deque(maxlen=history_size)
        
        self.current_frame: Optional[FrameMetrics] = None
        self.frame_count = 0
        
        self.stats = {
            'total_frames': 0,
            'frames_with_video': 0,
            'frames_with_features': 0,
            'frames_with_rag': 0,
            'frames_with_nn': 0,
            'frames_with_animation': 0,
            'avg_processing_time_ms': 0.0,
            'total_retrieved_memories': 0,
            'avg_relevance': 0.0,
        }
    
    def start_frame(self, frame_number: int) -> None:
        """Start monitoring a new frame."""
        self.current_frame = FrameMetrics(
            frame_number=frame_number,
            timestamp=time.time(),
            video_received=False
        )
        self.frame_count = frame_number
    
    def record_video_frame(self, frame: np.ndarray) -> None:
        """Record video frame reception and analysis."""
        if self.current_frame is None:
            self.logger.warning("No active frame context")
            return
        
        self.current_frame.video_received = True
        self.current_frame.video_shape = frame.shape
        self.current_frame.video_mean_intensity = float(np.mean(frame))
    
    def record_visual_features(self, features: np.ndarray) -> None:
        """Record visual feature extraction."""
        if self.current_frame is None:
            self.logger.warning("No active frame context")
            return
        
        self.current_frame.visual_features_extracted = True
        self.current_frame.visual_feature_dim = len(features)
        self.current_frame.visual_feature_mean = float(np.mean(features))
        self.current_frame.visual_feature_std = float(np.std(features))
    
    def record_rag_retrieval(self, rag_context: Dict[str, Any]) -> None:
        """Record RAG retrieval results."""
        if self.current_frame is None:
            self.logger.warning("No active frame context")
            return
        
        self.current_frame.rag_retrieved = True
        self.current_frame.rag_retrieved_count = rag_context.get('retrieved_count', 0)
        self.current_frame.rag_context_relevance = rag_context.get('context_relevance', 0.0)
        self.current_frame.rag_memory_types = rag_context.get('memory_types', [])
    
    def record_neural_network(self, output: np.ndarray) -> None:
        """Record neural network processing."""
        if self.current_frame is None:
            self.logger.warning("No active frame context")
            return
        
        self.current_frame.neural_network_processed = True
        self.current_frame.neural_network_output_dim = len(output)
        self.current_frame.neural_network_output_mean = float(np.mean(output))
    
    def record_animation_update(self, state: Dict[str, float]) -> None:
        """Record animation state update."""
        if self.current_frame is None:
            self.logger.warning("No active frame context")
            return
        
        self.current_frame.animation_updated = True
        self.current_frame.animation_state = state.copy()
    
    def end_frame(self) -> None:
        """End frame monitoring and update statistics."""
        if self.current_frame is None:
            self.logger.warning("No active frame context")
            return
        
        # Calculate processing time
        self.current_frame.processing_time_ms = (time.time() - self.current_frame.timestamp) * 1000.0
        
        # Store in history
        self.frame_metrics.append(self.current_frame)
        
        # Update statistics
        self._update_stats()
        
        # Log if verbose
        if self.verbose and self.frame_count % 30 == 0:
            self.logger.info(self.current_frame.to_string())
    
    def _update_stats(self) -> None:
        """Update aggregate statistics."""
        if not self.frame_metrics:
            return
        
        self.stats['total_frames'] = len(self.frame_metrics)
        self.stats['frames_with_video'] = sum(1 for m in self.frame_metrics if m.video_received)
        self.stats['frames_with_features'] = sum(1 for m in self.frame_metrics if m.visual_features_extracted)
        self.stats['frames_with_rag'] = sum(1 for m in self.frame_metrics if m.rag_retrieved)
        self.stats['frames_with_nn'] = sum(1 for m in self.frame_metrics if m.neural_network_processed)
        self.stats['frames_with_animation'] = sum(1 for m in self.frame_metrics if m.animation_updated)
        
        # Average processing time
        times = [m.processing_time_ms for m in self.frame_metrics]
        self.stats['avg_processing_time_ms'] = np.mean(times) if times else 0.0
        
        # Retrieve memories and relevance
        retrieved = [m.rag_retrieved_count for m in self.frame_metrics if m.rag_retrieved]
        self.stats['total_retrieved_memories'] = sum(retrieved)
        
        relevances = [m.rag_context_relevance for m in self.frame_metrics if m.rag_retrieved]
        self.stats['avg_relevance'] = np.mean(relevances) if relevances else 0.0
    
    def print_report(self) -> None:
        """Print comprehensive monitoring report."""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("COGNITIVE SYSTEM - MONITORING REPORT")
        self.logger.info("=" * 80)
        
        self.logger.info(f"Total Frames Processed: {self.stats['total_frames']}")
        self.logger.info(f"Average Processing Time: {self.stats['avg_processing_time_ms']:.2f}ms")
        self.logger.info("")
        
        self.logger.info("COMPONENT ACTIVATION:")
        total = self.stats['total_frames']
        if total > 0:
            self.logger.info(f"  Video Feed:         {self.stats['frames_with_video']:3d}/{total} ({100*self.stats['frames_with_video']/total:5.1f}%)")
            self.logger.info(f"  Feature Extraction: {self.stats['frames_with_features']:3d}/{total} ({100*self.stats['frames_with_features']/total:5.1f}%)")
            self.logger.info(f"  RAG Retrieval:      {self.stats['frames_with_rag']:3d}/{total} ({100*self.stats['frames_with_rag']/total:5.1f}%)")
            self.logger.info(f"  Neural Network:     {self.stats['frames_with_nn']:3d}/{total} ({100*self.stats['frames_with_nn']/total:5.1f}%)")
            self.logger.info(f"  Animation Update:   {self.stats['frames_with_animation']:3d}/{total} ({100*self.stats['frames_with_animation']/total:5.1f}%)")
        
        self.logger.info("")
        self.logger.info("RAG SYSTEM:")
        self.logger.info(f"  Total Memories Retrieved: {self.stats['total_retrieved_memories']}")
        self.logger.info(f"  Average Relevance Score: {self.stats['avg_relevance']:.3f}")
        
        self.logger.info("")
        self.logger.info("RECENT FRAMES:")
        for metric in list(self.frame_metrics)[-5:]:
            self.logger.info(f"  {metric.to_string()}")
        
        self.logger.info("=" * 80)
        self.logger.info("")
    
    def get_status_string(self) -> str:
        """Get brief status string."""
        if not self.frame_metrics:
            return "No frames recorded"
        
        last = self.frame_metrics[-1]
        video_status = "✓" if last.video_received else "✗"
        features_status = "✓" if last.visual_features_extracted else "✗"
        rag_status = "✓" if last.rag_retrieved else "✗"
        nn_status = "✓" if last.neural_network_processed else "✗"
        anim_status = "✓" if last.animation_updated else "✗"
        
        return f"[{video_status}V {features_status}F {rag_status}R {nn_status}N {anim_status}A] Frame {last.frame_number} ({last.processing_time_ms:.1f}ms)"
    
    def get_interaction_flow(self) -> str:
        """Get detailed interaction flow for debugging."""
        if not self.frame_metrics:
            return "No frames recorded"
        
        last = self.frame_metrics[-1]
        
        flow = []
        flow.append(f"Frame {last.frame_number} Processing Flow:")
        flow.append(f"  1. Video Input: {'YES ✓' if last.video_received else 'NO ✗'}")
        if last.video_received:
            flow.append(f"     └─ Shape: {last.video_shape}, Intensity: {last.video_mean_intensity:.3f}")
        
        flow.append(f"  2. Feature Extraction: {'YES ✓' if last.visual_features_extracted else 'NO ✗'}")
        if last.visual_features_extracted:
            flow.append(f"     └─ {last.visual_feature_dim}D, μ={last.visual_feature_mean:.3f}, σ={last.visual_feature_std:.3f}")
        
        flow.append(f"  3. RAG Retrieval: {'YES ✓' if last.rag_retrieved else 'NO ✗'}")
        if last.rag_retrieved:
            flow.append(f"     └─ Retrieved {last.rag_retrieved_count}, Relevance: {last.rag_context_relevance:.3f}")
            flow.append(f"     └─ Memory Types: {', '.join(last.rag_memory_types)}")
        
        flow.append(f"  4. Neural Network: {'YES ✓' if last.neural_network_processed else 'NO ✗'}")
        if last.neural_network_processed:
            flow.append(f"     └─ Output: {last.neural_network_output_dim}D, μ={last.neural_network_output_mean:.3f}")
        
        flow.append(f"  5. Animation Update: {'YES ✓' if last.animation_updated else 'NO ✗'}")
        if last.animation_updated and last.animation_state:
            for key, value in last.animation_state.items():
                flow.append(f"     └─ {key}: {value:.2f}")
        
        flow.append(f"  Total Time: {last.processing_time_ms:.2f}ms")
        
        return "\n".join(flow)
