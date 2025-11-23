# 🚀 LLM + RAG - Quick Start

## Install Local LLM (Choose One)

### Easiest: Ollama
```bash
# Download from https://ollama.ai
# Then run:
ollama run neural-chat
```

### Alternative: GPT4All
```bash
pip install gpt4all
# Then run (auto-downloads model)
```

## Run the Powerful System

```bash
python interactive_llm_rag.py
```

## Start Chatting

```
🤖 AI: Hello! I'm a powerful cognitive AI...

👤 You: Hi! Tell me about AI
🤖 AI [LLM]: Artificial Intelligence (AI) is the 
simulation of human intelligence by machines...

👤 You: What was I asking about?
🤖 AI [LLM]: You asked me about AI and I explained 
that it's the simulation of human intelligence...

👤 You: exit
```

## That's It!

- LLM provides intelligent responses
- RAG remembers everything
- Cognitive system adapts
- Everything local and private

## Tips

| Command | Action |
|---------|--------|
| Type text | Get intelligent LLM response |
| `status` | See system state |
| `memories` | View stored memories |
| `exit` | Quit |

## Modes

```bash
python interactive_llm_rag.py              # Full power
python interactive_llm_rag.py --no-llm     # RAG only (fast)
python interactive_llm_rag.py --speech     # With voice
```

## Performance

- **With LLM:** 1-10 seconds per response
- **Without LLM:** <1 second (RAG only)

Choose based on what you need:
- Want deep thinking? → Use with LLM
- Want fast responses? → Use `--no-llm`

---

**Enjoy your powerful local AI! 🎊**
