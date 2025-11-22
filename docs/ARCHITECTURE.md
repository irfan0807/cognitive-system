# Architecture Documentation

## System Overview

The Embodied Cognition System is designed around the principle that **virtual biology drives virtual cognition**. This document details how each component contributes to this integrated architecture.

## Core Principle: Biology → Cognition

Traditional AI systems process information in isolation from biological constraints. This system models how biological processes fundamentally shape cognitive functions:

```
Sensory Input → Brain Structures → Physiology → Neurotransmitters → Cognition → Behavior
                      ↑                                                    ↓
                      └────────────── Feedback Loop ──────────────────────┘
```

## Component Interactions

### 1. Sensory Processing Pipeline

**Input Sources:**
- Visual input (e.g., images, visual patterns)
- Auditory input (e.g., sounds, speech)
- Social context (presence/quality of social interaction)
- Threat detection (environmental dangers)
- Reward signals (positive outcomes)

**Processing Flow:**
```
Visual/Auditory Input
    ↓
Sensory Integration (VirtualNervousSystem)
    ↓
Feature Encoding (dimension reduction)
    ↓
Stored in Sensory Buffer
```

### 2. Brain Structure Processing

#### Brain Stem
- **Input:** Sensory intensity, stress level
- **Function:** Regulates basic life functions
- **Output:** 
  - Arousal level (0-1)
  - Heart rate (60-100 bpm)
  - Respiratory rate (10-20 bpm)
- **Effect on Cognition:** Higher arousal → increased attention and reactivity

#### Pituitary Gland
- **Input:** Social stimulus, positive emotions
- **Function:** Releases oxytocin (social bonding hormone)
- **Output:** Oxytocin level (0-1)
- **Effect on Cognition:** 
  - Increases trust and social approach behaviors
  - Buffers stress responses
  - Enhances memory of social events

#### Paraventricular Nucleus (PVN)
- **Input:** Threat level, arousal, oxytocin
- **Function:** Coordinates stress response
- **Output:** 
  - Stress response (0-1)
  - Cortisol signal (0-1)
- **Effect on Cognition:** 
  - High stress → avoidance behaviors
  - Modulated by oxytocin (social buffering)
  - Affects memory consolidation

### 3. Physiology Layer

#### Autonomic Nervous System
Balances two opposing systems:

**Sympathetic (Fight/Flight):**
- Activated by stress
- Increases heart rate
- Redirects resources to immediate action
- **Effect:** Narrows attention, increases reactivity

**Parasympathetic (Rest/Digest):**
- Activated by safety and social bonding
- Decreases heart rate
- Supports restoration
- **Effect:** Broadens attention, enables learning

#### Neurotransmitter System

**Dopamine:**
- **Function:** Reward processing, motivation
- **Increased by:** Rewards, positive outcomes
- **Effect on Cognition:** Motivation to repeat rewarded actions

**Serotonin:**
- **Function:** Mood regulation, social behavior
- **Increased by:** Social bonding, oxytocin
- **Decreased by:** Stress
- **Effect on Cognition:** Stable mood enables better decisions

**Norepinephrine:**
- **Function:** Attention, arousal
- **Increased by:** Stress, novelty
- **Effect on Cognition:** Focused attention on salient stimuli

**GABA:**
- **Function:** Inhibition, calmness
- **Effect on Cognition:** Reduces anxiety, enables deliberation

### 4. Cognitive Layer

#### Attention Network
```
Input: [Visual Features, Auditory Features, Emotion, Physiology]
    ↓
Neural Network (Real-time Learning)
    ↓
Attention Focus Vector
    ↓
Modulated by Norepinephrine Level
```

**Key Feature:** Attention is not a purely cognitive process—it's modulated by biological state (norepinephrine).

#### Decision Network
```
Input: [Attention Focus, Brain State, Physiology State, Emotion]
    ↓
Neural Network (Real-time Learning)
    ↓
Action Values
    ↓
Selected Action + Confidence
```

**Key Feature:** Decisions emerge from the interaction of cognitive processing (attention) and biological state (arousal, hormones, neurotransmitters).

### 5. Memory System

#### Memory Formation
Memories are formed when emotional significance exceeds a threshold:

```python
if abs(emotional_valence) > 0.3 or emotional_arousal > 0.3:
    form_memory()
```

#### Memory Structure
Each memory contains:
- **Visual features** (64-dimensional encoding)
- **Auditory features** (32-dimensional encoding)
- **Emotional valence** (-1 to +1: negative to positive)
- **Emotional arousal** (0 to 1: calm to excited)
- **Physiological state** (arousal, heart rate, oxytocin, etc.)
- **Context** (timestep, social context, threat level)
- **Strength** (0 to 1: weak to strong)

#### Memory Consolidation
Memory strength is computed based on emotional significance:

```
strength = base_strength + 
           emotional_weight * |valence| + 
           arousal_weight * arousal
```

This mimics biological memory consolidation, where emotionally significant events are remembered better.

#### Memory Retrieval
Retrieval uses cosine similarity across modalities:
- Visual similarity
- Auditory similarity
- Emotional similarity

Retrieved memories are weighted by their current strength.

## Example Scenarios

### Scenario A: Calm Social Interaction

**Input:**
- Visual: Mild (0.3)
- Auditory: Mild (0.3)
- Social: High (0.8)
- Threat: None (0.0)
- Reward: High (0.7)

**Biological Response:**
1. **Brain Stem:** Moderate arousal (~0.67)
2. **Pituitary:** High oxytocin (~0.84) from social stimulus
3. **PVN:** Low stress (~0.11) buffered by oxytocin
4. **Autonomic:** Parasympathetic dominant (rest/digest)
5. **Neurotransmitters:** High serotonin, moderate dopamine

**Cognitive Effects:**
- Broad, relaxed attention
- Positive mood enables exploration
- Social approach behaviors favored
- Memory formed with positive valence

### Scenario B: Threatening Situation

**Input:**
- Visual: Intense (0.8)
- Auditory: Intense (0.8)
- Social: None (0.0)
- Threat: High (0.9)
- Reward: None (0.0)

**Biological Response:**
1. **Brain Stem:** High arousal (~0.90)
2. **Pituitary:** Low oxytocin (no social support)
3. **PVN:** High stress (~0.39)
4. **Autonomic:** Sympathetic dominant (fight/flight)
5. **Neurotransmitters:** High norepinephrine, low serotonin

**Cognitive Effects:**
- Narrow, focused attention on threat
- Negative mood → avoidance behaviors
- Reduced learning (stress impairs consolidation)
- Strong memory formed (high arousal)

### Scenario C: Social Buffering

**Input:**
- Visual: Intense (0.7)
- Auditory: Intense (0.7)
- Social: High (0.9) - **key difference**
- Threat: High (0.8)
- Reward: Moderate (0.3)

**Biological Response:**
1. **Brain Stem:** High arousal (~0.89)
2. **Pituitary:** Very high oxytocin (~1.0) from social support
3. **PVN:** **Modulated stress** (~0.33) - much lower than Scenario B despite high threat!
4. **Autonomic:** More balanced (oxytocin buffers sympathetic)
5. **Neurotransmitters:** Serotonin remains higher due to social bonding

**Cognitive Effects:**
- Attention remains functional despite threat
- Better decision-making despite danger
- Social approach behaviors despite threat
- **Demonstrates how biology (oxytocin) directly modulates cognition (stress response)**

## Learning and Adaptation

### Real-time Learning
Both attention and decision networks learn continuously:

```python
# Each timestep
output = network.forward(input)
loss = compute_loss(output, target)
network.backward(gradient)  # Weights updated immediately
```

This enables:
- Adaptation to new environments
- Learning from immediate feedback
- No separate training/inference phases

### Memory-Based Learning
Past experiences influence future decisions:

```python
similar_memories = recall_similar_experiences(current_situation)
# Use similar memories to inform current decision
```

## Key Innovations

1. **Bidirectional Biology-Cognition Coupling**
   - Biology → Cognition: Physiology modulates attention and decisions
   - Cognition → Biology: Perceived threats/rewards affect physiology

2. **Social Buffering of Stress**
   - Oxytocin from social bonding reduces stress response
   - Models real neurobiological phenomenon

3. **Multimodal Memory Integration**
   - Single memory integrates emotion, vision, auditory
   - Mimics hippocampal memory formation

4. **Emotional Modulation of Consolidation**
   - Emotionally significant events remembered better
   - Matches biological memory systems

5. **Neurotransmitter-Mediated Cognition**
   - Specific neurotransmitters modulate specific functions
   - Dopamine → motivation
   - Serotonin → mood
   - Norepinephrine → attention
   - GABA → calmness

## Future Extensions

Potential enhancements to the system:

1. **Motor Control**
   - Add motor cortex and basal ganglia models
   - Enable embodied actions beyond decisions

2. **Circadian Rhythms**
   - Add time-of-day effects on arousal and neurotransmitters
   - Model sleep and its effects on memory consolidation

3. **Learning Rules**
   - Implement dopaminergic reward prediction error
   - Add hippocampal-cortical memory transfer

4. **Social Cognition**
   - Model theory of mind
   - Add empathy and emotional contagion

5. **Homeostatic Drives**
   - Add hunger, thirst, temperature regulation
   - Model how drives influence behavior

## Validation

The system has been validated through:
- Unit tests for each component (18 tests, all passing)
- Integration tests for complete system
- Scenario demonstrations showing expected emergent behaviors
- Verification that biology-cognition coupling produces realistic patterns

## References

This system draws inspiration from:
- Embodied cognition theory (Varela, Thompson, Rosch)
- Affective neuroscience (Panksepp, LeDoux)
- Predictive processing (Friston, Clark)
- Social neuroscience (Cacioppo, Decety)
