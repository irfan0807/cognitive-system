# Quick Start: Terminal Interface

This guide will help you get started with the terminal-based cognitive system interface.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/irfan0807/cognitive-system.git
cd cognitive-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For speech synthesis on Linux, install espeak:
```bash
sudo apt-get install espeak
```

## Running the App

### First Run (No Camera/Speech)
Test the interface without hardware:
```bash
python terminal_app.py --no-camera --no-speech
```

You should see a green terminal interface with:
- System state metrics
- Conversation area
- The AI will greet you

Press `Ctrl+C` to exit.

### With Camera Only
```bash
python terminal_app.py --no-speech
```

You'll see:
- Live camera feed as ASCII art
- The AI processing your video
- State updates based on what it sees

### Full Mode (Camera + Speech)
```bash
python terminal_app.py
```

The AI will:
- See you through the camera
- Talk to you using text-to-speech
- Display everything in the terminal

## What You'll See

```
╔═══════════════════════════════════════════════════════════════════╗
║  COGNITIVE SYSTEM - LIVE TERMINAL INTERFACE                       ║
╚═══════════════════════════════════════════════════════════════════╝
┏━━━━━━━━━━━━━━━━━━━━ 📹 Camera Feed ━━━━━━━━━━━━━━━━━━━━┓┏━━ 🧠 System State ━━┓
┃                                                         ┃┃ Arousal:    0.52   ┃
┃        [ASCII art of your face here]                    ┃┃ Mood:       0.48   ┃
┃                                                         ┃┃ Stress:     0.12   ┃
┃                                                         ┃┃ Heart Rate: 68 bpm ┃
┃                                                         ┃┃ Attention:  0.65   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛┗━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━ 💬 Conversation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 12:04:15 AI: Hello! I can see you through the camera.                       ┃
┃ 12:04:20 AI: I'm feeling quite alert and focused right now.                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Understanding the State Metrics

- **Arousal**: How alert/awake the system is (0.0 = calm, 1.0 = very alert)
- **Mood**: Emotional state (-1.0 = negative, 1.0 = positive)
- **Stress**: Current stress level (0.0 = relaxed, 1.0 = very stressed)
- **Heart Rate**: Virtual heart rate in beats per minute
- **Attention**: Focus level (0.0 = unfocused, 1.0 = highly focused)
- **Frame**: Current frame number

## Tips

1. **Terminal Size**: Maximize your terminal window for the best experience
2. **Green Theme**: The interface uses a retro green-on-black aesthetic
3. **Response Timing**: The AI speaks every 5 seconds based on its state
4. **Exit**: Press `Ctrl+C` anytime to exit cleanly

## Troubleshooting

### "Camera not available"
- Check if your camera is working: `ls /dev/video*`
- Make sure no other app is using the camera
- Run with `--no-camera` to skip camera

### "Speech engine error"
- On Linux: Install espeak (`sudo apt-get install espeak`)
- Run with `--no-speech` to disable speech

### Terminal looks weird
- Make sure you're using a modern terminal with true color support
- Try maximizing the window
- Use a terminal like iTerm2 (Mac), Windows Terminal, or Gnome Terminal (Linux)

## Next Steps

- Read the full [Terminal Interface README](TERMINAL_INTERFACE_README.md)
- Explore the main [README](README.md) for other demos
- Customize the AI's responses in `terminal_app.py`

## Need Help?

Check the logs at `/tmp/cognitive_terminal.log` if something goes wrong.
