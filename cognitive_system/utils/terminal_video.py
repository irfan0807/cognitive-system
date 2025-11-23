"""
Terminal Video Display Module

Displays video feed in terminal using ASCII art and Rich library.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TerminalVideoDisplay:
    """
    Displays video frames in terminal using ASCII art.
    """
    
    # ASCII characters from darkest to brightest
    ASCII_CHARS = " .:-=+*#%@"
    
    def __init__(self, width: int = 80, height: int = 40):
        """
        Initialize terminal video display.
        
        Args:
            width: Width in characters
            height: Height in characters
        """
        self.width = width
        self.height = height
        logger.info(f"Terminal video display initialized ({width}x{height})")
    
    def frame_to_ascii(self, frame: np.ndarray) -> str:
        """
        Convert video frame to ASCII art.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            ASCII art string
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to terminal dimensions
        resized = cv2.resize(gray, (self.width, self.height))
        
        # Convert to ASCII
        ascii_str = ""
        for row in resized:
            for pixel in row:
                # Map pixel value (0-255) to ASCII character
                char_index = int((pixel / 255.0) * (len(self.ASCII_CHARS) - 1))
                ascii_str += self.ASCII_CHARS[char_index]
            ascii_str += "\n"
        
        return ascii_str
    
    def frame_to_colored_ascii(self, frame: np.ndarray) -> list:
        """
        Convert video frame to colored ASCII art (for Rich library).
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of strings with color codes
        """
        # Resize frame
        resized = cv2.resize(frame, (self.width, self.height))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Create ASCII with colors
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                r, g, b = rgb_frame[y, x]
                # Calculate brightness
                brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
                char_index = int((brightness / 255.0) * (len(self.ASCII_CHARS) - 1))
                char = self.ASCII_CHARS[char_index]
                
                # For green terminal theme, we'll use green shades
                # Map brightness to green intensity
                green_intensity = int((brightness / 255.0) * 255)
                color_code = f"rgb(0,{green_intensity},0)"
                
                line += f"[{color_code}]{char}[/]"
            lines.append(line)
        
        return lines
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get terminal display dimensions.
        
        Returns:
            (width, height) in characters
        """
        return (self.width, self.height)
