"""
Vector Store for RAG System

Implements an efficient in-memory vector database for storing and retrieving
multimodal embeddings from video and audio streams.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class VectorEntry:
    """
    A single entry in the vector store.
    
    Attributes:
        id: Unique identifier for the entry
        embedding: The vector embedding
        metadata: Additional metadata (timestamp, modality, etc.)
        timestamp: When this entry was created
    """
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float


class VectorStore:
    """
    In-memory vector store for efficient similarity search.
    
    Uses cosine similarity for retrieval and supports multiple modalities
    (visual, auditory, multimodal).
    """
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.entries: List[VectorEntry] = []
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"VectorStore initialized with embedding_dim={embedding_dim}")
    
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any], 
            entry_id: Optional[str] = None, timestamp: Optional[float] = None) -> str:
        """
        Add a vector to the store.
        
        Args:
            embedding: The embedding vector to store
            metadata: Metadata associated with this embedding
            entry_id: Optional ID for the entry (auto-generated if not provided)
            timestamp: Optional timestamp (auto-generated if not provided)
            
        Returns:
            The ID of the added entry
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
        
        # Generate ID if not provided
        if entry_id is None:
            entry_id = f"entry_{len(self.entries)}"
        
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time()
        
        # Normalize the embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        entry = VectorEntry(
            id=entry_id,
            embedding=embedding,
            metadata=metadata,
            timestamp=timestamp
        )
        
        self.entries.append(entry)
        self.logger.debug(f"Added entry {entry_id} to vector store (total: {len(self.entries)})")
        
        return entry_id
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               modality_filter: Optional[str] = None,
               time_window: Optional[Tuple[float, float]] = None) -> List[Tuple[VectorEntry, float]]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            modality_filter: Optional filter by modality (e.g., 'visual', 'auditory')
            time_window: Optional time window (start, end) to filter entries
            
        Returns:
            List of (entry, similarity_score) tuples, sorted by similarity (highest first)
        """
        if len(self.entries) == 0:
            return []
        
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Filter entries based on criteria
        filtered_entries = self.entries
        
        if modality_filter is not None:
            filtered_entries = [
                e for e in filtered_entries 
                if e.metadata.get('modality') == modality_filter
            ]
        
        if time_window is not None:
            start_time, end_time = time_window
            filtered_entries = [
                e for e in filtered_entries
                if start_time <= e.timestamp <= end_time
            ]
        
        if len(filtered_entries) == 0:
            return []
        
        # Compute cosine similarities
        similarities = []
        for entry in filtered_entries:
            similarity = np.dot(query_embedding, entry.embedding)
            similarities.append((entry, float(similarity)))
        
        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_recent(self, n: int = 10, modality_filter: Optional[str] = None) -> List[VectorEntry]:
        """
        Get the most recent entries.
        
        Args:
            n: Number of recent entries to return
            modality_filter: Optional filter by modality
            
        Returns:
            List of recent entries
        """
        filtered_entries = self.entries
        
        if modality_filter is not None:
            filtered_entries = [
                e for e in filtered_entries
                if e.metadata.get('modality') == modality_filter
            ]
        
        # Sort by timestamp (most recent first)
        filtered_entries = sorted(filtered_entries, key=lambda x: x.timestamp, reverse=True)
        
        return filtered_entries[:n]
    
    def clear(self, modality: Optional[str] = None):
        """
        Clear entries from the store.
        
        Args:
            modality: If provided, only clear entries of this modality
        """
        if modality is None:
            self.entries.clear()
            self.logger.info("Cleared all entries from vector store")
        else:
            original_count = len(self.entries)
            self.entries = [
                e for e in self.entries
                if e.metadata.get('modality') != modality
            ]
            removed = original_count - len(self.entries)
            self.logger.info(f"Cleared {removed} entries with modality '{modality}'")
    
    def size(self) -> int:
        """Return the number of entries in the store."""
        return len(self.entries)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        modalities = {}
        for entry in self.entries:
            modality = entry.metadata.get('modality', 'unknown')
            modalities[modality] = modalities.get(modality, 0) + 1
        
        return {
            'total_entries': len(self.entries),
            'embedding_dim': self.embedding_dim,
            'modalities': modalities,
            'oldest_timestamp': min([e.timestamp for e in self.entries]) if self.entries else None,
            'newest_timestamp': max([e.timestamp for e in self.entries]) if self.entries else None
        }
