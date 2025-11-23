"""
Cognitive Visualizer for RAG System

Visualizes the cognitive behavior of the system during live video/audio processing,
showing attention, memory retrieval, emotional state, and decision-making in real-time.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging
import time


class CognitiveVisualizer:
    """
    Visualizes cognitive processes during live stream processing.
    
    This class generates text-based visualizations of:
    - Attention weights on different modalities
    - Memory retrieval results
    - Emotional/physiological state
    - Decision-making process
    - RAG context usage
    """
    
    def __init__(self, update_interval: float = 0.5):
        """
        Initialize the cognitive visualizer.
        
        Args:
            update_interval: Minimum time between visualization updates (seconds)
        """
        self.update_interval = update_interval
        self.last_update_time = 0.0
        self.logger = logging.getLogger(__name__)
        
        # Visualization history for trends
        self.attention_history = []
        self.arousal_history = []
        self.memory_usage_history = []
        
        self.logger.info("CognitiveVisualizer initialized")
    
    def should_update(self) -> bool:
        """
        Check if enough time has passed for a visualization update.
        
        Returns:
            True if visualization should be updated
        """
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
    
    def visualize_cognitive_state(self, state: Dict[str, Any], 
                                   stream_info: Optional[Dict[str, Any]] = None,
                                   rag_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a visualization of the current cognitive state.
        
        Args:
            state: Cognitive system state dictionary
            stream_info: Information about current video/audio streams
            rag_context: RAG retrieval context information
            
        Returns:
            Formatted visualization string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("COGNITIVE BEHAVIOR - LIVE FEED ANALYSIS")
        lines.append("=" * 70)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"Timestamp: {timestamp}")
        lines.append("")
        
        # Visualize attention
        if 'attention' in state:
            lines.append("ATTENTION DISTRIBUTION:")
            lines.extend(self._visualize_attention(state['attention']))
            lines.append("")
        
        # Visualize physiological state
        if 'physiology' in state:
            lines.append("PHYSIOLOGICAL STATE:")
            lines.extend(self._visualize_physiology(state['physiology']))
            lines.append("")
        
        # Visualize emotional state
        if 'emotion' in state:
            lines.append("EMOTIONAL STATE:")
            lines.extend(self._visualize_emotion(state['emotion']))
            lines.append("")
        
        # Visualize stream information
        if stream_info:
            lines.append("STREAM INFORMATION:")
            lines.extend(self._visualize_stream_info(stream_info))
            lines.append("")
        
        # Visualize RAG context
        if rag_context:
            lines.append("RAG CONTEXT RETRIEVAL:")
            lines.extend(self._visualize_rag_context(rag_context))
            lines.append("")
        
        # Visualize decision making
        if 'decision' in state:
            lines.append("DECISION MAKING:")
            lines.extend(self._visualize_decision(state['decision']))
            lines.append("")
        
        # Visualize memory usage
        if 'memory_usage' in state:
            lines.append("MEMORY ACTIVITY:")
            lines.extend(self._visualize_memory_usage(state['memory_usage']))
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _visualize_attention(self, attention: Dict[str, float]) -> List[str]:
        """Visualize attention weights."""
        lines = []
        
        modalities = ['visual', 'auditory', 'memory', 'internal']
        for modality in modalities:
            if modality in attention:
                weight = attention[modality]
                bar = self._create_bar(weight, width=30)
                lines.append(f"  {modality:12s}: {bar} {weight:5.2f}")
        
        # Track history
        if 'visual' in attention and 'auditory' in attention:
            total_attention = attention['visual'] + attention['auditory']
            self.attention_history.append(total_attention)
            if len(self.attention_history) > 50:
                self.attention_history.pop(0)
        
        return lines
    
    def _visualize_physiology(self, physiology: Dict[str, float]) -> List[str]:
        """Visualize physiological state."""
        lines = []
        
        # Heart rate
        if 'heart_rate' in physiology:
            hr = physiology['heart_rate']
            hr_norm = (hr - 60) / 60  # Normalize around resting rate
            bar = self._create_bar(hr_norm, width=30, min_val=0.0, max_val=1.0)
            lines.append(f"  Heart Rate  : {bar} {hr:5.1f} bpm")
        
        # Arousal
        if 'arousal' in physiology:
            arousal = physiology['arousal']
            bar = self._create_bar(arousal, width=30)
            lines.append(f"  Arousal     : {bar} {arousal:5.2f}")
            
            self.arousal_history.append(arousal)
            if len(self.arousal_history) > 50:
                self.arousal_history.pop(0)
        
        # Stress
        if 'stress' in physiology:
            stress = physiology['stress']
            bar = self._create_bar(stress, width=30)
            lines.append(f"  Stress      : {bar} {stress:5.2f}")
        
        # Energy
        if 'energy' in physiology:
            energy = physiology['energy']
            bar = self._create_bar(energy, width=30)
            lines.append(f"  Energy      : {bar} {energy:5.2f}")
        
        return lines
    
    def _visualize_emotion(self, emotion: Dict[str, float]) -> List[str]:
        """Visualize emotional state."""
        lines = []
        
        if 'valence' in emotion:
            valence = emotion['valence']
            # Center at 0.5 for neutral
            bar = self._create_bar(valence, width=30)
            sentiment = "Positive" if valence > 0.5 else "Negative" if valence < 0.5 else "Neutral"
            lines.append(f"  Valence     : {bar} {valence:5.2f} ({sentiment})")
        
        if 'arousal' in emotion:
            arousal = emotion['arousal']
            bar = self._create_bar(arousal, width=30)
            intensity = "High" if arousal > 0.7 else "Medium" if arousal > 0.3 else "Low"
            lines.append(f"  Arousal     : {bar} {arousal:5.2f} ({intensity})")
        
        if 'mood' in emotion:
            mood = emotion['mood']
            bar = self._create_bar(mood, width=30)
            lines.append(f"  Mood        : {bar} {mood:5.2f}")
        
        return lines
    
    def _visualize_stream_info(self, stream_info: Dict[str, Any]) -> List[str]:
        """Visualize stream information."""
        lines = []
        
        if 'video_frame_count' in stream_info:
            lines.append(f"  Video Frames   : {stream_info['video_frame_count']}")
        
        if 'audio_chunk_count' in stream_info:
            lines.append(f"  Audio Chunks   : {stream_info['audio_chunk_count']}")
        
        if 'video_features' in stream_info:
            feat = stream_info['video_features']
            if isinstance(feat, np.ndarray):
                activity = np.abs(feat).mean()
                bar = self._create_bar(activity, width=20)
                lines.append(f"  Visual Activity: {bar} {activity:5.2f}")
        
        if 'audio_features' in stream_info:
            feat = stream_info['audio_features']
            if isinstance(feat, np.ndarray):
                activity = np.abs(feat).mean()
                bar = self._create_bar(activity, width=20)
                lines.append(f"  Audio Activity : {bar} {activity:5.2f}")
        
        return lines
    
    def _visualize_rag_context(self, rag_context: Dict[str, Any]) -> List[str]:
        """Visualize RAG retrieval context."""
        lines = []
        
        if 'retrieved_count' in rag_context:
            lines.append(f"  Retrieved Memories: {rag_context['retrieved_count']}")
        
        if 'top_similarities' in rag_context:
            similarities = rag_context['top_similarities']
            lines.append(f"  Top Similarities:")
            for i, sim in enumerate(similarities[:3], 1):
                bar = self._create_bar(sim, width=20)
                lines.append(f"    #{i}: {bar} {sim:5.3f}")
        
        if 'context_relevance' in rag_context:
            relevance = rag_context['context_relevance']
            bar = self._create_bar(relevance, width=30)
            lines.append(f"  Context Relevance: {bar} {relevance:5.3f}")
        
        if 'memory_types' in rag_context:
            lines.append(f"  Memory Types: {', '.join(rag_context['memory_types'])}")
        
        return lines
    
    def _visualize_decision(self, decision: Dict[str, Any]) -> List[str]:
        """Visualize decision-making process."""
        lines = []
        
        if 'action' in decision:
            lines.append(f"  Selected Action: {decision['action']}")
        
        if 'confidence' in decision:
            confidence = decision['confidence']
            bar = self._create_bar(confidence, width=30)
            lines.append(f"  Confidence     : {bar} {confidence:5.3f}")
        
        if 'alternatives' in decision:
            lines.append(f"  Alternatives   : {', '.join(decision['alternatives'][:3])}")
        
        return lines
    
    def _visualize_memory_usage(self, memory_usage: Dict[str, Any]) -> List[str]:
        """Visualize memory system activity."""
        lines = []
        
        if 'retrievals' in memory_usage:
            retrievals = memory_usage['retrievals']
            lines.append(f"  Retrievals: {retrievals}")
            
            self.memory_usage_history.append(retrievals)
            if len(self.memory_usage_history) > 50:
                self.memory_usage_history.pop(0)
        
        if 'stored' in memory_usage:
            lines.append(f"  Stored    : {memory_usage['stored']}")
        
        if 'total_memories' in memory_usage:
            lines.append(f"  Total     : {memory_usage['total_memories']}")
        
        return lines
    
    def _create_bar(self, value: float, width: int = 30, 
                    min_val: float = 0.0, max_val: float = 1.0) -> str:
        """
        Create a text-based bar visualization.
        
        Args:
            value: Value to visualize
            width: Width of the bar in characters
            min_val: Minimum value for scaling
            max_val: Maximum value for scaling
            
        Returns:
            Bar string
        """
        # Clip value to range
        value = max(min_val, min(max_val, value))
        
        # Normalize to 0-1
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        
        # Create bar
        filled = int(normalized * width)
        empty = width - filled
        
        bar = "█" * filled + "░" * empty
        return f"[{bar}]"
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics from visualization history.
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {}
        
        if len(self.attention_history) > 0:
            stats['avg_attention'] = np.mean(self.attention_history)
            stats['attention_trend'] = 'increasing' if self.attention_history[-1] > self.attention_history[0] else 'decreasing'
        
        if len(self.arousal_history) > 0:
            stats['avg_arousal'] = np.mean(self.arousal_history)
            stats['arousal_stability'] = 1.0 - np.std(self.arousal_history)
        
        if len(self.memory_usage_history) > 0:
            stats['total_memory_retrievals'] = sum(self.memory_usage_history)
        
        return stats
