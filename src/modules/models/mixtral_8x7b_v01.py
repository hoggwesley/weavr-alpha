from together import Together
from modules.config_loader import load_api_key, get_system_prompt
import re
import traceback

# Enable debugging - commented out for production
DEBUG_MODE = False  # Change to True for debugging

def debug_print(message):
    """Print debug information when debug mode is enabled"""
    if DEBUG_MODE:
        print(f"🔍 DEBUG [mixtral]: {message}")

def generate_response(prompt):
    """Handles Mixtral-8x7B AI requests with proper instruction formatting."""
    # debug_print(f"generate_response called with prompt (first 100 chars): {prompt[:100]}...")
    
    try:
        api_key = load_api_key()
        client = Together(api_key=api_key)

        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        # debug_print(f"Using model: {model_name}")

        temperature = 0.65
        max_tokens = 2048  # Increase max tokens for longer responses
        
        # Add stop sequences to ensure proper formatting
        stop_sequences = ["[INST]", "[/INST]", "User:", "Assistant:", "Human:", "AI:"]
        
        response = client.completions.create(
            model=model_name,
            prompt=prompt.strip(),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.7,
            stop=stop_sequences,
            repetition_penalty=1.15  # Prevent format deviations
        )

        raw_text = response.choices[0].text.strip()
        token_count = len(raw_text.split())
        # debug_print(f"Received {token_count} tokens in response")

        # Add distinct debug prints
        # if "Generate detailed reasoning steps" in prompt:
        #     debug_print("Generating REASONING STEPS")
        # elif "Synthesize a concise and well-structured final answer" in prompt:
        #     debug_print("Generating FINAL ANSWER")

        return raw_text, token_count
        
    except Exception as e:
        # debug_print(f"Error in generate_response: {str(e)}")
        # debug_print(traceback.format_exc())
        return f"An error occurred: {str(e)}", 0
