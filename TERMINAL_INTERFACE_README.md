# Terminal Interface for Cognitive System

A retro-style terminal interface where the AI can see you through the camera and talk to you through text-to-speech.

## Features

- **Live Video Feed**: The system sees you through your camera (displayed as ASCII art in the terminal)
- **Text-to-Speech**: The AI talks to you using voice synthesis
- **Green Terminal Theme**: Classic green-on-black terminal aesthetic
- **Real-time State Display**: Shows the AI's arousal, mood, stress, heart rate, and attention levels
- **Conversation History**: Displays recent conversation in the terminal

## Requirements

```bash
pip install -r requirements.txt
```

Additional dependencies:
- `rich` - Terminal UI library
- `pyttsx3` - Text-to-speech engine
- `opencv-python` - Camera capture

## Usage

### Full Mode (Camera + Speech)
```bash
python terminal_app.py
```

### Camera Only (No Speech)
```bash
python terminal_app.py --no-speech
```

### Text Only (No Camera)
```bash
python terminal_app.py --no-camera
```

### Quiet Mode (No Camera, No Speech)
```bash
python terminal_app.py --no-camera --no-speech
```

## Terminal Layout

```
╔═══════════════════════════════════════════════════════════════════╗
║  COGNITIVE SYSTEM - LIVE TERMINAL INTERFACE                       ║
╚═══════════════════════════════════════════════════════════════════╝
┏━━━━━━━━━━━━━━━━━━━━ 📹 Camera Feed ━━━━━━━━━━━━━━━━━━━━┓┏━━ 🧠 System State ━━┓
┃                                                         ┃┃ Arousal:    0.52   ┃
┃                  ASCII VIDEO HERE                       ┃┃ Mood:       0.48   ┃
┃                                                         ┃┃ Stress:     0.12   ┃
┃                                                         ┃┃ Heart Rate: 68 bpm ┃
┃                                                         ┃┃ Attention:  0.65   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛┗━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━ 💬 Conversation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 12:04:15 AI: Hello! I can see you through the camera.                       ┃
┃ 12:04:20 AI: I'm feeling quite alert and focused right now.                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## How It Works

1. **Camera Input**: The system captures video frames from your camera
2. **Visual Processing**: Frames are converted to ASCII art for terminal display
3. **Feature Extraction**: Visual features are extracted and fed to the cognitive system
4. **State Processing**: The cognitive system processes the input and updates its internal state
5. **AI Responses**: Based on its state, the AI generates contextual responses
6. **Speech Output**: Responses are spoken using text-to-speech
7. **Display Update**: The terminal interface updates in real-time (4 FPS)

## Keyboard Controls

- `Ctrl+C`: Exit the application

## Configuration

You can modify the following in `terminal_app.py`:

- `speech_interval`: Time between AI responses (default: 5 seconds)
- `video_display` dimensions: Terminal video width/height (default: 60x20)
- `tts_engine` rate: Speech rate in words per minute (default: 150)
- Terminal refresh rate (default: 4 FPS)

## Troubleshooting

### Camera Not Working
- Check if your camera is available: `ls /dev/video*`
- Try running with `--no-camera` flag
- Ensure no other application is using the camera

### Speech Not Working
- On Linux, you may need to install espeak: `sudo apt-get install espeak`
- On macOS, the built-in speech engine should work
- Try running with `--no-speech` flag to disable

### Terminal Display Issues
- Ensure your terminal supports true color
- Try maximizing your terminal window for better display
- Some terminals may not support all Rich features

## Architecture

```
terminal_app.py
├── TerminalCognitiveApp (main class)
│   ├── Initialize cognitive system
│   ├── Initialize TTS engine
│   ├── Initialize camera
│   └── Run main loop
│       ├── Capture video frame
│       ├── Convert to ASCII
│       ├── Extract visual features
│       ├── Update cognitive system
│       ├── Generate AI response
│       ├── Speak response
│       └── Update display
└── Components
    ├── cognitive_system.utils.terminal_video (ASCII video)
    ├── cognitive_system.utils.tts_engine (speech)
    └── rich library (terminal UI)
```

## Credits

Built on top of the Cognitive System framework with embodied AI and deep neural networks.
