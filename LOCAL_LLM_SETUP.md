# 🤖 Local LLM Integration Guide

## What This Does

Connects your cognitive system to a **local language model** for:
- ✅ More powerful, intelligent responses
- ✅ Contextual understanding
- ✅ Better conversations
- ✅ Combined with RAG memory for maximum power
- ✅ Complete privacy (runs locally)

## Quick Start

```bash
python interactive_llm_rag.py
```

Or without LLM (RAG only):
```bash
python interactive_llm_rag.py --no-llm
```

## Setup Local LLM

Choose one option:

### Option 1: Ollama (Recommended - Easiest)

**Install Ollama:**
- Download from [ollama.ai](https://ollama.ai)
- Install and follow setup

**Run Ollama:**
```bash
ollama run neural-chat
# or
ollama run mistral
# or
ollama run llama2
```

Then run your cognitive system:
```bash
python interactive_llm_rag.py
```

**Available Models:**
- `neural-chat` - Good for chat (fast)
- `mistral` - Fast and capable
- `llama2` - Powerful but slower
- `orca-mini-3b` - Lightweight

### Option 2: GPT4All (Easy, No Setup)

**Install:**
```bash
pip install gpt4all
```

**Run:**
```bash
python interactive_llm_rag.py
```

The system will download a model automatically.

### Option 3: Llama.cpp (Advanced)

**Install:**
```bash
pip install llama-cpp-python
```

**Get a model:**
```bash
# Download from huggingface
# e.g., mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

**Configure in code:**
Edit `interactive_llm_rag.py` and set model path.

### Option 4: Hugging Face Transformers

**Install:**
```bash
pip install transformers torch
```

**Run:**
```bash
python interactive_llm_rag.py
```

Uses DistilGPT2 by default.

## Recommended Setup

**For best results, use Ollama:**

1. Download and install Ollama from https://ollama.ai
2. Run: `ollama run neural-chat`
3. In another terminal: `python interactive_llm_rag.py`

That's it! Full local LLM + RAG + Cognitive System.

## How It Works

```
You Type: "Hello, how are you?"
    ↓
Cognitive System processes input
    ↓
RAG retrieves relevant memories
    ↓
LLM generates intelligent response using context
    ↓
Response marked [LLM] showing power
    ↓
You see response immediately
```

## Example Conversation

```
🤖 AI: Hello! I'm a powerful cognitive AI powered by 
a local language model and retrieval-augmented memory...

👤 You: Tell me about machine learning
🤖 AI [LLM]: Machine learning is a subset of artificial 
intelligence that focuses on teaching computers to learn 
from data without being explicitly programmed...

👤 You: What is RAG?
🤖 AI [LLM]: RAG, or Retrieval-Augmented Generation, 
combines information retrieval with generation models 
to provide more accurate and contextual responses...

👤 You: Remember this conversation
🤖 AI [LLM]: I've stored our discussion about machine 
learning and RAG in my memory. I can reference these 
concepts in future conversations.
```

## Modes

```bash
# Full mode (LLM + RAG + Cognitive)
python interactive_llm_rag.py

# With speech
python interactive_llm_rag.py --speech

# RAG only (no LLM)
python interactive_llm_rag.py --no-llm

# RAG only with speech
python interactive_llm_rag.py --no-llm --speech
```

## Commands

| Command | Action |
|---------|--------|
| Type anything | Send to AI (responds with LLM power) |
| `status` | Show system state + LLM status |
| `memories` | Show what was learned |
| `context` | Show current context |
| `exit` | Quit |

## System Features

✅ **LLM Integration:** Multiple LLM backends supported
✅ **RAG Memory:** Recalls past conversations
✅ **Cognitive System:** Embodied cognition + neural networks
✅ **Context Building:** Combines state + memories + history
✅ **Response Markers:** [LLM] shows model-generated responses
✅ **Fallback System:** Works without LLM using RAG
✅ **Privacy:** Everything runs locally
✅ **Learning:** Improves with each conversation

## Supported LLM Backends

| Backend | Install | Model |
|---------|---------|-------|
| Ollama | Easy | Many choices |
| GPT4All | `pip install gpt4all` | Auto-download |
| Llama.cpp | `pip install llama-cpp-python` | GGUF files |
| Transformers | `pip install transformers` | Hugging Face |

## Performance

- **LLM response time:** 1-10 seconds (depends on model)
- **RAG fallback:** <1 second
- **Memory search:** <1ms
- **Cognitive processing:** <100ms

## Troubleshooting

**LLM not responding?**
```bash
# Check if Ollama is running
ollama list

# Or install GPT4All
pip install gpt4all
```

**Slow responses?**
- Use a smaller model (`neural-chat`, `mistral`)
- Use `--no-llm` for faster RAG-only responses

**LLM not detected?**
```bash
# The system will automatically fall back to RAG
# You'll see [RAG] responses instead of [LLM]
python interactive_llm_rag.py
```

**Want different model?**
- For Ollama: `ollama run mistral` or `ollama run llama2`
- For GPT4All: Models download automatically
- For Llama.cpp: Point to your GGUF file

## What You Get

✨ **More Powerful AI:**
- Understands context better
- Generates longer, more detailed responses
- Explains concepts clearly
- Has personality and empathy

✨ **With RAG Memory:**
- Remembers what you discussed
- References past conversations
- Learns from interactions
- Gets smarter over time

✨ **With Cognitive System:**
- Tracks emotional state
- Adapts responses based on arousal/mood
- Makes decisions with behavior engine
- Operates like a real mind

## Files

- `interactive_llm_rag.py` - Main app with LLM integration
- `/tmp/cognitive_llm.log` - Debug logs

## Next Steps

1. Install Ollama or GPT4All
2. Run: `python interactive_llm_rag.py`
3. Type: `Hello!`
4. Watch intelligent responses

## Full Power Unlocked 🚀

You now have:
- ✅ Local LLM for intelligence
- ✅ RAG for memory
- ✅ Cognitive system for reasoning
- ✅ Everything runs locally
- ✅ Complete privacy

Enjoy! 🎊
