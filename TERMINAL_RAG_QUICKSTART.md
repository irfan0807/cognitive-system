# Quick Start: Terminal RAG App

## Installation

Ensure you have the required packages:

```bash
pip install rich opencv-python pyttsx3 numpy torch scipy librosa sounddevice matplotlib
```

## Run It Now

```bash
# Full interactive mode
python terminal_app_rag.py

# Or without camera (text only)
python terminal_app_rag.py --no-camera
```

## What You'll See

```
╔═══════════════════════════════════════════════╗
║  COGNITIVE SYSTEM - TERMINAL WITH RAG        ║
╚═══════════════════════════════════════════════╝

[Camera feed in ASCII] | [System state display]

📊 RAG Memories: 0 | Relevance: 0%

💬 Conversation
12:04:15 AI: Hello! I can see you through...

You: _
```

## How to Interact

1. **Type a message** and press Enter
2. **System responds** using RAG-based reasoning
3. **Type `exit`** to quit
4. **Watch the RAG indicator** show memory retrieval

## Example Conversation

```
You: What do you see?
AI [RAG]: I can see you through the camera.

You: Remember this moment
AI [RAG]: That's new to me! I'm storing this experience 
in my memory. It will help me learn and grow.

You: What was I talking about?
AI [RAG]: Based on my memory, I can connect your comment 
to 2 relevant past experiences with 65% confidence.
```

## Features Enabled

✓ Live camera feed (ASCII art)
✓ Text-to-speech responses
✓ RAG-based memory and decision-making
✓ Interactive chat
✓ Real-time system state monitoring
✓ Green retro terminal theme

## Try Different Modes

```bash
# No camera (fast, text-only)
python terminal_app_rag.py --no-camera

# No speech (silent)
python terminal_app_rag.py --no-speech

# No RAG (simple responses only)
python terminal_app_rag.py --no-rag

# Everything minimal
python terminal_app_rag.py --no-camera --no-speech --no-rag
```

## What's Special About RAG?

The system uses **Retrieval-Augmented Generation** which means:

1. **It remembers**: Every interaction stored as embeddings
2. **It retrieves**: When you chat, it finds relevant memories
3. **It decides**: Responses based on actual past experiences
4. **It learns**: Each interaction makes it smarter

## Keyboard Shortcuts

- Type your message → Send to AI
- `exit` or `quit` → Close app
- `Ctrl+C` → Force quit

## See What's Happening

Watch the logs:
```bash
tail -f /tmp/cognitive_terminal_rag.log
```

---

**That's it! Start chatting with your cognitive AI now! 🚀**
