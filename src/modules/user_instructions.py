"""
Module for managing persistent user instructions for the AI.
"""
import os
import json
from pathlib import Path

# Default location for storing user instructions
DEFAULT_INSTRUCTIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "user_instructions.json")

def get_instructions_path():
    """Get the path to the user instructions file."""
    instructions_dir = os.path.dirname(DEFAULT_INSTRUCTIONS_PATH)
    os.makedirs(instructions_dir, exist_ok=True)
    return DEFAULT_INSTRUCTIONS_PATH

def load_instructions():
    """Load user instructions from disk."""
    instructions_path = get_instructions_path()
    if not os.path.exists(instructions_path):
        return ""
        
    try:
        with open(instructions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("instructions", "")
    except Exception as e:
        print(f"Error loading user instructions: {str(e)}")
        return ""

def save_instructions(instructions):
    """Save user instructions to disk."""
    instructions_path = get_instructions_path()
    try:
        with open(instructions_path, 'w', encoding='utf-8') as f:
            json.dump({"instructions": instructions}, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving user instructions: {str(e)}")
        return False

def get_combined_prompt(system_prompt, user_instructions):
    """Combine system prompt with user instructions."""
    if not user_instructions:
        return system_prompt
        
    # Add user instructions after system prompt
    return f"{system_prompt}\n\nUser-Provided Instructions:\n{user_instructions}"
