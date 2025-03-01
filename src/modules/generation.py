import sys
import os

# ✅ Fix path so 'modules' is recognized correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.config_loader import load_api_key, get_model_name, get_model_api_name
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
    """Routes AI requests to the appropriate model handler based on config.yaml."""
    model_key = get_model_name()  # ✅ This is now 'mixtral_8x7b_v01' or 'qwen_72b_instruct'
    model_name = get_model_api_name()  # ✅ This is 'mistralai/Mixtral-8x7B-Instruct-v0.1'

    if not model_name:
        raise ValueError(f"ERROR: Model '{model_key}' is not found in config.yaml!")

    module_name = f"modules.models.{model_key}"  # ✅ Uses correct Python module name

    try:
        model_module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"ERROR: Model module '{module_name}' not found. Ensure it's implemented in 'src/modules/models/'.")

    if not hasattr(model_module, "generate_response"):
        raise AttributeError(f"ERROR: Module '{module_name}' is missing the 'generate_response' function!")

    response_text, token_count = model_module.generate_response(query, context, task_type)

    return clean_response(response_text), token_count

if __name__ == "__main__":
    print("🔹 Running AI Generation Test...")
    response = query_together("What is Weavr AI?")
    print(f"✅ AI Response:\n{response}")
