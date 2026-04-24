#!/usr/bin/env python3
"""
Persona Engine Integration Example

Demonstrates how to integrate the emotion system into a simple chatbot.
This is a minimal standalone example — for full integration with Hermes Agent,
see INTEGRATION.md.

Usage:
    python integration_example.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_state_manager import EmotionStateManager


def main():
    # Setup paths
    hermes_home = os.environ.get('HERMES_HOME', os.path.expanduser('~/.hermes'))
    
    # Check for required files
    soul_path = Path(hermes_home) / 'SOUL.md'
    if not soul_path.exists():
        print(f"[!] SOUL.md not found at {soul_path}")
        print("    Copy SOUL.template.md to ~/.hermes/SOUL.md and customize it.")
        return
    
    # Initialize emotion system
    print("[*] Initializing emotion system...")
    manager = EmotionStateManager(hermes_home=hermes_home)
    print("[OK] Emotion system ready.")
    
    # Apply time decay (simulates gap between conversations)
    manager.apply_time_decay_if_needed()
    
    # Show current state
    state = manager._read_state()
    emotion = state['frontmatter'].get('emotion_state', {})
    print(f"\n[State] aff={emotion.get('affection', 60)} "
          f"tru={emotion.get('trust', 60)} "
          f"pos={emotion.get('possessiveness', 60)} "
          f"pat={emotion.get('patience', 60)}")
    
    # Conversation loop
    messages = []
    print("\nType messages to test emotion detection. Ctrl+C to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        
        # Add to message history
        messages.append({"role": "user", "content": user_input})
        
        # Step 1: Detect emotion triggers
        event = manager.detector.detect_emotion_event(messages)
        
        if event:
            print(f"  [Trigger] {event.trigger_type} "
                  f"(confidence: {event.confidence:.2f})")
            print(f"  [Deltas]  {event.deltas}")
            
            # Step 2: Update emotion state
            manager.update_emotion_state(
                messages=messages,
                trigger_event=event
            )
            
            # Step 3: Show updated state
            state = manager._read_state()
            emotion = state['frontmatter'].get('emotion_state', {})
            print(f"  [State]   aff={emotion.get('affection', 60)} "
                  f"tru={emotion.get('trust', 60)} "
                  f"pos={emotion.get('possessiveness', 60)} "
                  f"pat={emotion.get('patience', 60)}")
        else:
            print("  [No trigger detected]")
        
        # Step 4: Generate tone modifier
        tone = manager.generate_tone_modifier()
        if tone:
            # Truncate for display
            display = tone[:200] + "..." if len(tone) > 200 else tone
            print(f"  [Tone]    {display}")
        
        print()


if __name__ == "__main__":
    main()
