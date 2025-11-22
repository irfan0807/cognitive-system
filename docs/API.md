# API Reference

## Main Classes

### EmbodiedCognitionSystem

Main integration class that brings together all components.

```python
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem

system = EmbodiedCognitionSystem(
    visual_dim=64,          # Dimensionality of visual processing
    auditory_dim=32,        # Dimensionality of auditory processing
    attention_dim=128,      # Dimensionality of attention mechanism
    learning_rate=0.01      # Learning rate for neural networks
)
```

#### Methods

##### `process_sensory_input()`
Process sensory input through the complete embodied cognition system.

```python
state = system.process_sensory_input(
    visual_input: np.ndarray,      # Visual sensory input
    auditory_input: np.ndarray,    # Auditory sensory input
    social_context: float = 0.0,   # Social interaction (0-1)
    threat_level: float = 0.0,     # Perceived threat (0-1)
    reward_signal: float = 0.0     # Reward/positive reinforcement (0-1)
) -> CognitiveState
```

Returns a `CognitiveState` object containing:
- `brain_state`: Dictionary of brain structure states
- `physiological_state`: Dictionary of physiological states
- `cognitive_modulation`: Dictionary of neurotransmitter effects
- `memory_stats`: Statistics about memory system
- `attention_focus`: Attention vector
- `emotional_state`: Tuple of (valence, arousal)

##### `make_decision()`
Make a decision based on current cognitive state.

```python
action, confidence = system.make_decision(
    possible_actions: List[str]    # List of possible action names
) -> Tuple[str, float]
```

Returns:
- `action`: Selected action (string)
- `confidence`: Confidence in the decision (0-1)

##### `get_state_summary()`
Get a summary of current system state.

```python
summary = system.get_state_summary() -> Dict[str, any]
```

Returns a dictionary with keys:
- `timestep`: Current timestep
- `arousal`: Arousal level
- `heart_rate`: Heart rate (bpm)
- `oxytocin`: Oxytocin level
- `stress`: Stress response
- `energy`: Energy level
- `homeostasis`: Homeostatic balance
- `attention_level`: Attention level
- `mood`: Mood level
- `motivation`: Motivation level
- `emotional_valence`: Emotional valence
- `emotional_arousal`: Emotional arousal
- `total_memories`: Number of memories

##### `recall_similar_experiences()`
Retrieve similar experiences from memory.

```python
memories = system.recall_similar_experiences(
    visual_query: Optional[np.ndarray] = None,
    auditory_query: Optional[np.ndarray] = None,
    emotional_query: Optional[Tuple[float, float]] = None,
    top_k: int = 3
) -> List[Tuple[MultimodalMemory, float]]
```

Returns list of (memory, similarity_score) tuples.

##### `learn_from_feedback()`
Learn from action outcomes using real-time learning.

```python
system.learn_from_feedback(
    previous_input: Dict[str, np.ndarray],  # Previous sensory input
    action_taken: str,                      # Action that was taken
    outcome_reward: float                   # Reward received
)
```

---

### VirtualBrain

Integrates brain structures (brain stem, pituitary, PVN).

```python
from cognitive_system.brain import VirtualBrain

brain = VirtualBrain()
```

#### Methods

##### `process()`
Process inputs through all brain structures.

```python
state = brain.process(
    sensory_input: float = 0.0,      # Sensory stimulation (0-1)
    social_stimulus: float = 0.0,    # Social interaction (0-1)
    threat_level: float = 0.0,       # Perceived threat (0-1)
    positive_emotion: float = 0.0    # Positive emotion (0-1)
) -> Dict[str, float]
```

Returns dictionary with:
- `arousal`: Arousal level
- `heart_rate`: Heart rate (bpm)
- `respiratory_rate`: Respiratory rate (bpm)
- `oxytocin`: Oxytocin level
- `stress_response`: Stress response level
- `cortisol_signal`: Cortisol signal

---

### RealtimeNeuralNetwork

Neural network with online learning.

```python
from cognitive_system.neural import RealtimeNeuralNetwork

network = RealtimeNeuralNetwork(
    layer_sizes=[10, 20, 10],    # List of layer sizes
    learning_rate=0.01           # Learning rate
)
```

#### Methods

##### `forward()`
Forward pass through the network.

```python
output = network.forward(x: np.ndarray) -> np.ndarray
```

##### `train_online()`
Perform online learning with a single sample.

```python
loss = network.train_online(
    x: np.ndarray,        # Input sample
    target: np.ndarray    # Target output
) -> float
```

Returns the loss value.

---

### MultimodalMemorySystem

Memory system integrating emotion, visual, and auditory stimuli.

```python
from cognitive_system.memory import MultimodalMemorySystem

memory = MultimodalMemorySystem(
    visual_dim=64,           # Visual feature dimension
    auditory_dim=32,         # Auditory feature dimension
    max_memories=1000,       # Maximum memories to store
    decay_rate=0.01          # Memory decay rate
)
```

#### Methods

##### `form_memory()`
Form a new multimodal memory.

```python
memory_obj = memory.form_memory(
    visual_input: np.ndarray,                          # Visual stimulus
    auditory_input: np.ndarray,                        # Auditory stimulus
    emotional_valence: float,                          # Valence (-1 to 1)
    emotional_arousal: float,                          # Arousal (0 to 1)
    physiological_state: Optional[Dict] = None,        # Physiological state
    context: Optional[Dict] = None                     # Context info
) -> MultimodalMemory
```

##### `retrieve_similar()`
Retrieve memories similar to query.

```python
memories = memory.retrieve_similar(
    query_visual: Optional[np.ndarray] = None,
    query_auditory: Optional[np.ndarray] = None,
    query_emotion: Optional[Tuple[float, float]] = None,
    top_k: int = 5
) -> List[Tuple[MultimodalMemory, float]]
```

##### `get_statistics()`
Get memory system statistics.

```python
stats = memory.get_statistics() -> Dict[str, any]
```

Returns:
- `total_memories`: Number of memories
- `avg_strength`: Average memory strength
- `avg_emotional_valence`: Average emotional valence
- `avg_arousal`: Average arousal
- `strongest_memory_strength`: Strongest memory
- `weakest_memory_strength`: Weakest memory

---

### VirtualPhysiology

Complete virtual physiology system.

```python
from cognitive_system.physiology import VirtualPhysiology

physiology = VirtualPhysiology()
```

#### Methods

##### `update()`
Update complete physiological state.

```python
state = physiology.update(
    brain_state: Dict[str, float],              # Brain state
    sensory_inputs: Dict[str, np.ndarray],      # Sensory inputs
    reward: float = 0.0                         # Reward signal
) -> Dict[str, any]
```

Returns:
- `nervous_system_state`: NervousSystemState object
- `energy_level`: Current energy level
- `homeostatic_balance`: Homeostatic balance
- `cognitive_modulation`: Dictionary of cognitive modulation factors

##### `rest()`
Simulate rest/recovery period.

```python
physiology.rest(duration: float = 1.0)
```

---

## Data Classes

### CognitiveState

Represents complete cognitive state.

**Attributes:**
- `brain_state`: Dict[str, float]
- `physiological_state`: Dict[str, any]
- `cognitive_modulation`: Dict[str, float]
- `memory_stats`: Dict[str, any]
- `attention_focus`: np.ndarray
- `emotional_state`: Tuple[float, float]

### MultimodalMemory

Represents a single multimodal memory.

**Attributes:**
- `timestamp`: datetime
- `emotional_valence`: float (-1 to 1)
- `emotional_arousal`: float (0 to 1)
- `visual_features`: np.ndarray
- `auditory_features`: np.ndarray
- `physiological_state`: Dict[str, float]
- `context`: Dict[str, any]
- `strength`: float (0 to 1)

### NervousSystemState

Represents nervous system state.

**Attributes:**
- `sympathetic_activity`: float (0 to 1)
- `parasympathetic_activity`: float (0 to 1)
- `neurotransmitter_levels`: Dict[str, float]
- `sensory_integration`: np.ndarray

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from cognitive_system.embodied_cognition import EmbodiedCognitionSystem

# Initialize system
system = EmbodiedCognitionSystem()

# Process sensory input
visual = np.random.randn(64)
auditory = np.random.randn(32)

state = system.process_sensory_input(
    visual_input=visual,
    auditory_input=auditory,
    social_context=0.8,
    threat_level=0.1,
    reward_signal=0.6
)

# Get state summary
summary = system.get_state_summary()
print(f"Arousal: {summary['arousal']:.3f}")
print(f"Mood: {summary['mood']:.3f}")

# Make decision
actions = ["approach", "avoid", "explore", "rest"]
action, confidence = system.make_decision(actions)
print(f"Decision: {action} (confidence: {confidence:.3f})")
```

### Learning from Experience

```python
# Process situation
state = system.process_sensory_input(visual, auditory)

# Make decision
action, confidence = system.make_decision(["approach", "avoid"])

# Get feedback
outcome_reward = 1.0 if action == "approach" else -0.5

# Learn from outcome
system.learn_from_feedback(
    previous_input={'visual': visual, 'auditory': auditory},
    action_taken=action,
    outcome_reward=outcome_reward
)
```

### Memory Retrieval

```python
# Recall similar positive experiences
positive_memories = system.recall_similar_experiences(
    emotional_query=(0.8, 0.5),  # Positive valence, moderate arousal
    top_k=3
)

for memory, similarity in positive_memories:
    print(f"Valence: {memory.emotional_valence:.2f}")
    print(f"Arousal: {memory.emotional_arousal:.2f}")
    print(f"Similarity: {similarity:.3f}")
```

### Using Individual Components

```python
from cognitive_system.brain import VirtualBrain
from cognitive_system.memory import MultimodalMemorySystem

# Use brain independently
brain = VirtualBrain()
brain_state = brain.process(
    sensory_input=0.7,
    social_stimulus=0.5,
    threat_level=0.2
)

# Use memory system independently
memory = MultimodalMemorySystem(visual_dim=64, auditory_dim=32)
mem = memory.form_memory(
    visual_input=np.random.randn(64),
    auditory_input=np.random.randn(32),
    emotional_valence=0.8,
    emotional_arousal=0.6
)
```
