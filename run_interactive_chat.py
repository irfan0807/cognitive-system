#!/usr/bin/env python3
"""
Quick test of the interactive RAG chat system
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interactive_rag_chat import InteractiveTerminalCognitive

# Create app
print("Starting Cognitive System...")
app = InteractiveTerminalCognitive(use_camera=False, use_speech=False)

try:
    app.initialize()
    app.chat()
except KeyboardInterrupt:
    print("\n\nExiting...")
finally:
    app.cleanup()
