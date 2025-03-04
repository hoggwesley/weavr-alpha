from together import Together
from modules.config_loader import load_api_key
import re
import traceback

# Enable debug mode
DEBUG_MODE = True

def debug_print(message):
    """Print debug information when debug mode is enabled"""
    if DEBUG_MODE:
        print(f"🔍 DEBUG [mistral-7b]: {message}")

def generate_response(prompt):
    """Handles Mistral-7B AI requests with proper instruction formatting."""
    debug_print(f"generate_response called with prompt (first 100 chars): {prompt[:100]}...")
    
    try:
        api_key = load_api_key()
        client = Together(api_key=api_key)

        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        debug_print(f"Using model: {model_name}")

        temperature = 0.7  # Slightly higher temperature for more creative reasoning
        max_tokens = 1024  # Reduced for reasoning steps
        
        # Add stop sequences to ensure proper formatting
        stop_sequences = ["[INST]", "[/INST]", "User:", "Assistant:", "Human:", "AI:"]
        
        response = client.completions.create(
            model=model_name,
            prompt=prompt.strip(),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.7,
            stop=stop_sequences,
            repetition_penalty=1.1
        )

        raw_text = response.choices[0].text.strip()
        token_count = len(raw_text.split())
        debug_print(f"Received {token_count} tokens in response")
        debug_print("Generating REASONING STEPS with Mistral-7B")
        
        return raw_text, token_count
        
    except Exception as e:
        debug_print(f"Error in generate_response: {str(e)}")
        debug_print(traceback.format_exc())
        return f"An error occurred: {str(e)}", 0
