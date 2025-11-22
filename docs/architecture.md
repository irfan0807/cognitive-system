# System Architecture

## Overview

The Cognitive System implements an autonomous embodied AI framework centered on the principle that **virtual biology drives virtual cognition and real-time learning and experience**.

## Architectural Layers

### 1. Embodied Cognition Core

The `EmbodiedCognitionSystem` class serves as the central orchestrator that integrates all system components into a coherent whole.

**Key Responsibilities:**
- Integration of virtual biology, neural networks, memory, and behavior
- Real-time simulation loop management
- Experience formation and learning coordination
- System state management

**Design Principle:**
The system is **centered on the virtual body** - all cognition emerges from the body's interaction with the environment, not from abstract symbolic processing.

### 2. Virtual Biology Layer

This layer implements the character's physiological substrate.

#### Components:

**VirtualNervousSystem:**
- Whole nervous system simulation
- Physiological parameter regulation (heart rate, breathing, stress)
- Chemical regulation (oxytocin, cortisol, dopamine)
- Motor state management
- Stimulus response (e.g., music beat synchronization)

**VirtualBrain:**
- **Brainstem:** Motor control, autonomic functions
  - Oculomotor nuclei for eye movements
  - Respiratory and cardiovascular centers
  - Motor pathway coordination
  
- **Pituitary Gland:** Hormone release
  - Oxytocin (social bonding, stress reduction)
  - ACTH (stress response)
  - Vasopressin
  
- **Paraventricular Nucleus:** Stress and autonomic regulation
  - Coordinates with pituitary gland
  - Processes emotional and physiological inputs

**Physiological Response Example:**
```
Stress Detected
    ↓
Paraventricular Nucleus Activates
    ↓
Pituitary Releases ACTH/Cortisol
    ↓
Heart Rate Increases (60 → 120 bpm)
    ↓
Breathing Rate Increases
```

### 3. Neural Processing Layer

**NeuralNetworkController:**
- Simplified brain simulation using neural networks
- Forward processing: sensory + physiological → motor commands
- Real-time learning and adaptation
- Hebbian-style weight updates based on experience

**Information Flow:**
```
Sensory Input + Physiological State
    ↓
Input Encoding (22-dimensional vector)
    ↓
Hidden Layer (50 neurons, tanh activation)
    ↓
Output Layer (20 neurons)
    ↓
Motor Commands + Attention + Arousal
```

### 4. Memory Layer

**MultimodalMemorySystem:**

Implements the requirement that memories are reconstructed **multimodally**:

Each memory integrates:
1. **Emotional modality:** Valence (positive/negative), arousal, stress
2. **Visual modality:** Visual features, attention points
3. **Auditory modality:** Auditory features, intensity
4. **Physiological modality:** Heart rate, hormones
5. **Behavioral modality:** Actions being performed

**Memory Operations:**
- **Storage:** Automatic during experience formation
- **Consolidation:** Strong emotional experiences strengthen memories
- **Decay:** Gradual weakening over time
- **Recall:** Multimodal similarity matching
- **Reconsolidation:** Retrieved memories are strengthened

**Example - Emotional Conditioning:**
```
Word "fire" presented
    ↓
Negative emotional response (valence = -0.9)
    ↓
Memory formed integrating:
  - Visual: Flames, orange/red colors
  - Auditory: Word "fire" encoding
  - Emotion: High negative valence, high arousal
  - Physiology: Elevated heart rate, high cortisol
    ↓
Future encounters with "fire" recall this multimodal memory
    ↓
Conditioned response: Automatic stress/fear response
```

### 5. Behavior Layer

**BehaviorEngine:**

Generates all character behaviors in real time based on:
- Neural network outputs
- Physiological state
- Environmental context

**Behavior Categories:**

1. **Motor Behaviors:**
   - Position commands
   - Movement intensity (modulated by arousal)
   - Stress-induced tremor

2. **Expressive Behaviors:**
   - Facial expressions (6 basic emotions)
   - Vocal tone (pitch, volume, rate)
   - Mapped from emotional state

3. **Attention Behaviors:**
   - Gaze targeting
   - Head orientation
   - Focus strength

4. **Environmental Interactions:**
   - Video watching (attention allocation)
   - Game playing (action selection)
   - Music response (rhythmic movement)
   - Object manipulation

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT                               │
│  Visual | Auditory | Tactile | Social | Environmental       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Sensory Processing   │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │      Virtual Nervous System            │
        │  - Stress regulation                   │
        │  - Heart rate / breathing              │
        │  - Chemical levels (hormones)          │
        └────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │        Virtual Brain                   │
        │  ┌──────────────────────────────────┐ │
        │  │  Paraventricular Nucleus (PVN)   │ │
        │  │  - Stress response               │ │
        │  │  - Signals to pituitary          │ │
        │  └────────┬─────────────────────────┘ │
        │           │                            │
        │  ┌────────▼─────────────────────────┐ │
        │  │  Pituitary Gland                 │ │
        │  │  - Oxytocin release              │ │
        │  │  - ACTH release                  │ │
        │  └────────┬─────────────────────────┘ │
        │           │                            │
        │  ┌────────▼─────────────────────────┐ │
        │  │  Brainstem                       │ │
        │  │  - Oculomotor nuclei             │ │
        │  │  - Autonomic control             │ │
        │  │  - Motor pathways                │ │
        │  └──────────────────────────────────┘ │
        └────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │   Neural Network Controller            │
        │  - Encode inputs                       │
        │  - Forward pass                        │
        │  - Generate responses                  │
        │  - Learn from experience               │
        └────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │      Behavior Engine                   │
        │  - Motor behaviors                     │
        │  - Expressive behaviors                │
        │  - Attention allocation                │
        │  - Environmental interaction           │
        └────────────┬───────────────────────────┘
                     │
                     ├────────────────────────────┐
                     │                            │
                     ▼                            ▼
        ┌────────────────────────┐   ┌───────────────────────┐
        │  Multimodal Memory     │   │   Motor Output        │
        │  - Store experience    │   │   to Environment      │
        │  - Consolidate         │   └───────────────────────┘
        │  - Enable recall       │
        └────────────────────────┘
```

## Key Design Patterns

### 1. Embodied Cognition Pattern

**Principle:** Cognition emerges from body-environment interaction

**Implementation:**
- All sensory processing includes physiological context
- Motor commands are modulated by physiological state
- Memories include full body state (not just abstract concepts)

### 2. Real-Time Processing Pattern

**Principle:** Character experiences in real-time, not batch processing

**Implementation:**
- Continuous simulation loop with delta-time updates
- Smooth physiological transitions (not instantaneous changes)
- Neural network forward passes every frame
- Experience formation is continuous

### 3. Multimodal Integration Pattern

**Principle:** All experiences integrate multiple modalities

**Implementation:**
- Memories bind visual + auditory + emotional + physiological
- Recall can use any modality as cue
- Learning strengthens cross-modal associations

### 4. Biological Plausibility Pattern

**Principle:** System mirrors biological organization

**Implementation:**
- Hierarchical brain structure (brainstem → higher areas)
- Chemical regulation through hormone systems
- Autonomic nervous system controls physiology
- Emotional states emerge from physiological changes

## Scalability Considerations

### Current Implementation
- Single character simulation
- Simplified neural networks
- Basic sensory encoding
- Moderate memory capacity

### Future Extensions
1. **Advanced Neural Models:**
   - Replace simple feedforward nets with recurrent/transformer architectures
   - Implement more detailed cortical areas
   - Add working memory systems

2. **Enhanced Sensory Processing:**
   - Computer vision for real visual input
   - Audio processing for real sound
   - Multimodal sensor fusion

3. **Social Cognition:**
   - Multi-character interactions
   - Theory of mind
   - Social learning

4. **Long-term Memory:**
   - Hierarchical memory organization
   - Sleep/consolidation cycles
   - Forgetting and interference

## Performance Characteristics

**Update Cycle (60 FPS target):**
- Physiological update: ~0.5ms
- Brain processing: ~1ms
- Neural network forward: ~2ms
- Behavior generation: ~0.5ms
- Memory storage: ~0.5ms
- **Total: ~5ms (well under 16ms budget)**

**Memory Scaling:**
- Memory per experience: ~2KB
- 10,000 experiences: ~20MB
- Recall time: O(n) where n = number of memories
- Optimization: Indexing, clustering for large-scale

## Summary

The architecture implements a cohesive embodied AI system where:

1. **Virtual biology** provides the substrate (physiology, brain)
2. **Neural networks** process and learn in real-time
3. **Multimodal memory** captures rich experiences
4. **Behavior emerges** from the interaction of all components
5. **Experience drives learning** in continuous feedback loop

This creates a truly autonomous character that experiences its world rather than merely processing abstract symbols.
