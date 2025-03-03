import sys
import os

# ✅ Fix path so 'modules' is recognized correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.config_loader import load_api_key, get_model_name, get_model_api_name, get_system_prompt  # ✅ Now properly imported

import importlib
import tiktoken
import re

def count_tokens(text):
    """Returns the token count for a given text using OpenAI-compatible tokenizer."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def clean_response(response_text):
    """Removes unwanted meta-text from AI responses."""
    patterns = [
        r"User Satisfaction:.*",
        r"System Improvement Suggestions:.*",
    ]
    
    for pattern in patterns:
        response_text = re.sub(pattern, "", response_text, flags=re.MULTILINE)

    return response_text.strip()

def query_together(query, context="", task_type="default"):
    """Routes AI requests while ensuring clean response formatting."""
    model_key = get_model_name()  
    model_name = get_model_api_name()

    if not model_name:
        raise ValueError(f"ERROR: Model '{model_key}' is not found in config.yaml!")

    module_name = f"modules.models.{model_key}"  # ✅ Correct path

    try:
        model_module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"ERROR: Model module '{module_name}' not found.")

    if not hasattr(model_module, "generate_response"):
        raise AttributeError(f"ERROR: Module '{module_name}' is missing the 'generate_response' function!")

    # ✅ Enforce system prompt for stricter behavior
    system_prompt = get_system_prompt()

    # ✅ Get AI response (with optional CoT reasoning)
    response_text, token_count, reasoning_steps = model_module.generate_response(
        query, context, task_type
    )

    # ✅ Stronger cleanup for hallucinated dialogue
    response_text = response_text.replace("User Query:", "").replace("--- AI Response ---", "").strip()
    response_text = response_text.replace("Context:", "").replace("User:", "").replace("Response:", "").strip()

    return response_text, token_count, reasoning_steps

if __name__ == "__main__":
    print("🔹 Running AI Generation Test...")
    response, token_count, reasoning_steps = query_together("What is Weavr AI?", task_type="cot")
    print(f"✅ AI Response:\n{response}")

    if reasoning_steps:
        print("\n🔍 Chain-of-Thought Reasoning:")
        for step in reasoning_steps:
            print(f" - {step}")
