#!/usr/bin/env python3
"""
Stress Control Demo

Shows how to interactively control stress levels in the animation.

Usage:
    python3 stress_demo.py --interactive    # Interactive mode with keyboard control
    python3 stress_demo.py --scenario       # Run stress scenarios
"""

import sys
sys.path.insert(0, '.')

import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_control():
    """Demonstrate basic stress control."""
    from cognitive_system.utils.stress_controller import StressController
    
    logger.info("=" * 70)
    logger.info("STRESS CONTROLLER - BASIC DEMO")
    logger.info("=" * 70)
    logger.info("")
    
    controller = StressController(initial_stress=0.4, step_size=0.15)
    
    logger.info(f"Initial: {controller.get_status_string()}")
    logger.info("")
    
    logger.info("Increasing stress 3 times...")
    for i in range(3):
        controller.increase_stress()
        logger.info(f"  Step {i+1}: {controller.get_status_string()}")
    
    logger.info("")
    logger.info("Decreasing stress 2 times...")
    for i in range(2):
        controller.decrease_stress()
        logger.info(f"  Step {i+1}: {controller.get_status_string()}")
    
    logger.info("")
    logger.info("Setting specific levels...")
    for level in [0.0, 0.3, 0.7, 1.0]:
        controller.set_stress(level)
        logger.info(f"  Set to {level}: {controller.get_status_string()}")
    
    logger.info("")
    logger.info("✓ Demo complete")


def demo_presets():
    """Demonstrate preset stress levels."""
    from cognitive_system.utils.stress_controller import StressController, StressLevel
    
    logger.info("=" * 70)
    logger.info("STRESS CONTROLLER - PRESETS DEMO")
    logger.info("=" * 70)
    logger.info("")
    
    controller = StressController()
    
    presets = [
        StressLevel.VERY_LOW,
        StressLevel.LOW,
        StressLevel.NORMAL,
        StressLevel.MODERATE,
        StressLevel.HIGH,
        StressLevel.VERY_HIGH,
    ]
    
    for preset in presets:
        controller.set_preset(preset)
        logger.info(f"{preset.name:12} ({preset.value:.1f}): {controller.get_status_string()}")
    
    logger.info("")
    logger.info("✓ Demo complete")


def demo_animation_bridge():
    """Demonstrate stress-animation bridge."""
    from cognitive_system.utils.stress_controller import StressController, StressAnimationBridge
    
    logger.info("=" * 70)
    logger.info("STRESS-ANIMATION BRIDGE DEMO")
    logger.info("=" * 70)
    logger.info("")
    
    controller = StressController(initial_stress=0.4)
    bridge = StressAnimationBridge(controller)
    
    # Base animation state
    base_state = {
        'arousal': 0.5,
        'stress': 0.4,
        'mood': 0.7,
        'heart_rate': 60.0,
        'attention': 0.5,
        'emotional_valence': 0.0,
        'emotional_arousal': 0.5,
        'oxytocin': 0.5
    }
    
    logger.info("SCENARIO 1: Low Stress State")
    controller.set_stress(0.2)
    modified = bridge.get_animation_state_with_stress(base_state)
    logger.info(f"  Stress: {modified['stress']:.2f}")
    logger.info(f"  Arousal: {modified['arousal']:.2f}")
    logger.info(f"  Heart Rate: {modified['heart_rate']:.0f} bpm")
    logger.info(f"  Mood: {modified['mood']:.2f}")
    logger.info(f"  Attention: {modified['attention']:.2f}")
    
    logger.info("")
    logger.info("SCENARIO 2: High Stress State")
    controller.set_stress(0.8)
    modified = bridge.get_animation_state_with_stress(base_state)
    logger.info(f"  Stress: {modified['stress']:.2f}")
    logger.info(f"  Arousal: {modified['arousal']:.2f}")
    logger.info(f"  Heart Rate: {modified['heart_rate']:.0f} bpm")
    logger.info(f"  Mood: {modified['mood']:.2f}")
    logger.info(f"  Attention: {modified['attention']:.2f}")
    
    logger.info("")
    logger.info("✓ Demo complete")


def demo_interactive():
    """Interactive stress control demo."""
    from cognitive_system.utils.stress_controller import InteractiveStressController
    
    logger.info("=" * 70)
    logger.info("INTERACTIVE STRESS CONTROLLER")
    logger.info("=" * 70)
    logger.info("")
    
    controller = InteractiveStressController(initial_stress=0.4, step_size=0.1)
    controller.print_controls()
    
    logger.info("Enter stress commands (or 'quit' to exit):")
    logger.info("")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting...")
                break
            
            # Parse numeric input
            try:
                value = float(user_input)
                controller.set_stress(value)
                logger.info(controller.get_status_string())
                continue
            except ValueError:
                pass
            
            # Handle key input
            if user_input:
                result = controller.handle_key_press(user_input)
                if result is not None:
                    logger.info(controller.get_status_string())
                else:
                    logger.info(f"Unknown command: {user_input}")
        
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    logger.info("✓ Interactive demo complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress Control Demo")
    parser.add_argument("--basic", action="store_true", help="Run basic control demo")
    parser.add_argument("--presets", action="store_true", help="Run presets demo")
    parser.add_argument("--bridge", action="store_true", help="Run animation bridge demo")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    
    args = parser.parse_args()
    
    # If no specific demo, run all
    if not any([args.basic, args.presets, args.bridge, args.interactive]):
        args.basic = True
        args.presets = True
        args.bridge = True
    
    if args.basic:
        demo_basic_control()
        logger.info("")
    
    if args.presets:
        demo_presets()
        logger.info("")
    
    if args.bridge:
        demo_animation_bridge()
        logger.info("")
    
    if args.interactive:
        demo_interactive()
