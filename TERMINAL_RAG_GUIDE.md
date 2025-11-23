# Terminal Cognitive System with RAG Integration

## Overview

This is an advanced interactive terminal application that enables you to interact with the cognitive system through:

- **📹 Live Video**: Camera feed displayed as ASCII art in the terminal
- **🎤 Audio Output**: Text-to-speech responses from the AI
- **💬 Chat Interface**: Type messages to interact with the system
- **🧠 RAG System**: The AI uses a Retrieval-Augmented Generation (RAG) system to:
  - Remember past interactions
  - Retrieve relevant memories
  - Make informed decisions based on context
  - Learn and grow from conversations

## Key Features

### Interactive Chat
- Type messages directly in the terminal
- System responds with context-aware answers
- All interactions are stored in memory
- Type `exit`, `quit`, or `bye` to close the application

### RAG-Powered Decision Making
The system uses the RAG (Retrieval-Augmented Generation) system to:
1. **Store Memories**: Every frame and interaction is stored as embeddings
2. **Retrieve Context**: When you chat, the system retrieves relevant past memories
3. **Generate Responses**: Responses are based on actual context and memories
4. **Mark Decisions**: AI responses show `[RAG]` when using memory-based decisions

### Real-time Monitoring
- **System State**: Arousal, mood, stress, heart rate, attention
- **RAG Status**: Number of retrieved memories and relevance score
- **Conversation History**: Last 5 messages displayed in real-time
- **Frame Counter**: Visual feedback on processing

### Green Terminal Aesthetic
- Retro green-on-black theme
- 60x20 character ASCII video display
- 4 FPS terminal refresh rate
- Rich formatting with borders and panels

## Usage

### Basic Usage (Full Mode)
```bash
python terminal_app_rag.py
```
Starts the system with:
- Camera feed
- Text-to-speech
- RAG system enabled
- Chat interface ready

### Specific Modes

```bash
# Camera only (no speech)
python terminal_app_rag.py --no-speech

# Text-only (no camera)
python terminal_app_rag.py --no-camera

# Without RAG system
python terminal_app_rag.py --no-rag

# Quiet mode (no camera, no speech)
python terminal_app_rag.py --no-camera --no-speech

# Text-only without RAG
python terminal_app_rag.py --no-camera --no-rag
```

## Terminal Layout

```
╔═════════════════════════════════════════════════╗
║  COGNITIVE SYSTEM - TERMINAL WITH RAG         ║
╚═════════════════════════════════════════════════╝

┌─────────────────────────┐ ┌──────────────────┐
│ 📹 Camera Feed          │ │ 🧠 System State  │
│                         │ │ Arousal:    0.52 │
│  [ASCII video frame]    │ │ Mood:       0.48 │
│                         │ │ Stress:     0.12 │
│                         │ │ Heart Rate: 68   │
│                         │ │ Attention:  0.65 │
└─────────────────────────┘ └──────────────────┘

╭──────────────────────────────────────────────╮
│ 🧠 RAG System                                │
│ RAG Memories: 42 | Relevance: 85%            │
╰──────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 💬 Conversation                            ┃
┃ 12:04:15 AI: Hello! How can I help?        ┃
┃ 12:04:20 User: Tell me about yourself      ┃
┃ 12:04:22 AI [RAG]: Based on my memory...   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

You: _
```

## How RAG Decision Making Works

### When You Chat

1. **Input Processing**: Your text input is converted to an embedding
2. **Memory Retrieval**: The system searches its memory for related past experiences
3. **Context Evaluation**: 
   - **High Relevance (>60%)**: Strong connection to past memories
   - **Medium Relevance (30-60%)**: Some related memories found
   - **Low Relevance (<30%)**: New or unrelated experience
4. **Response Generation**: The system generates a response based on:
   - Retrieved memories and their types (visual, auditory, etc.)
   - Relevance score
   - Current emotional state
   - Number of memories found

### Example Conversations

**Example 1: Familiar Topic**
```
You: Tell me about coffee
AI [RAG]: Based on 12 memories from visual, auditory modalities, 
I'm 78% confident: That relates to what I've learned!
```

**Example 2: Partially Familiar**
```
You: How do you feel about mountains?
AI [RAG]: Interesting! I found 5 related memories. 
This connects to my past with 42% certainty.
```

**Example 3: Novel Topic**
```
You: What is a sentient nebula?
AI: That's new to me! I'm storing this experience in my memory. 
It will help me learn and grow.
```

## System Components

### Cognitive System
- **Embodied Cognition**: Processes sensory inputs
- **Virtual Nervous System**: Maintains physiological state (arousal, stress)
- **Neural Network**: 22 inputs → 128 hidden → 32 outputs
- **Behavior Engine**: Decides actions based on state

### RAG System
- **Vector Store**: Stores embeddings of experiences (128-dimensional)
- **Video Processor**: Extracts features from camera frames
- **Audio Processor**: Processes audio input
- **Retrieval Engine**: Finds relevant memories using similarity search

### Interface
- **Terminal Display**: Rich UI with real-time updates (4 FPS)
- **ASCII Video**: Converts camera frames to ASCII art
- **Text-to-Speech**: Speaks responses aloud
- **Input Handler**: Processes user chat input

## Technical Details

### Dependencies
- `rich>=13.0.0` - Terminal UI
- `pyttsx3>=2.90` - Text-to-speech
- `opencv-python>=4.8.0` - Camera input
- `torch>=2.0.0` - Neural networks
- `numpy` - Numerical computing

### Integration Architecture

```
User Input
    ↓
Terminal Interface
    ↓
    ├→ Chat Processor → RAG System → Response Generator
    ├→ Camera Feed → Video Processor → RAG System
    └→ State Monitor → Display Update
        ↓
    Text-to-Speech Output
```

### Memory Storage
- Each frame stores 128-dimensional embeddings
- Metadata includes: modality, timestamp, intensity
- Retrieval uses cosine similarity
- Top-K retrieval (default: 3 memories)

## Commands

During runtime, you can type:

| Command | Effect |
|---------|--------|
| Any text | Send to system for processing |
| `exit` | Close application |
| `quit` | Close application |
| `bye` | Close application |

## Monitoring

Check the logs for detailed information:
```bash
tail -f /tmp/cognitive_terminal_rag.log
```

## Performance Notes

- **Frame Rate**: 4 FPS terminal refresh
- **Processing Time**: ~0.5-0.6ms per frame
- **Memory**: Embeddings stored up to system limit
- **Responsiveness**: User input processed in real-time

## Troubleshooting

### Camera Not Working
```bash
python terminal_app_rag.py --no-camera
```

### Speech Not Working
```bash
python terminal_app_rag.py --no-speech
```

### RAG System Issues
```bash
python terminal_app_rag.py --no-rag
```

### Full Text Mode
```bash
python terminal_app_rag.py --no-camera --no-speech
```

## Future Enhancements

1. **Voice Input**: Speech-to-text for natural conversation
2. **Custom Responses**: More sophisticated response generation models
3. **Memory Persistence**: Save/load memories across sessions
4. **Adjustable Settings**: Terminal size, refresh rate, colors
5. **Extended RAG**: Multi-hop reasoning over memories
6. **Emotion Tracking**: Display emotional state changes
7. **Memory Visualization**: Show retrieved memory connections
8. **Long-term Learning**: Semantic memory consolidation

## Notes

- The system learns from every interaction with you
- Camera input is processed in real-time but not stored (only embeddings)
- All responses marked with [RAG] are memory-augmented decisions
- The system is privacy-aware and doesn't store raw video
- Audio responses are generated on-the-fly

---

**Enjoy interacting with your cognitive system! 🧠💬**
