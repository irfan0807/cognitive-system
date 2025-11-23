#!/bin/bash
# Run the Cognitive System App with Animation and Video Feed

cd "$(dirname "$0")"

echo "=========================================="
echo "Cognitive System - Animation + Video Feed"
echo "=========================================="
echo ""

# Check if camera flag is present
if [ "$1" == "--test" ] || [ "$1" == "--no-video" ]; then
    echo "Running in TEST mode (no camera required)"
    python3 app_with_video.py --test
else
    echo "Running with LIVE VIDEO FEED"
    echo "(Make sure your camera is connected)"
    echo ""
    python3 app_with_video.py
fi
