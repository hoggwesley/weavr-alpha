from together import Together
from modules.config_loader import load_api_key  # 🔹 Import API key loader

import tiktoken  # Needed for token counting

def count_tokens(text):
    """Returns the token count for a given text using OpenAI's tokenizer."""
    encoding = tiktoken.get_encoding("cl100k_base")  # Compatible with OpenAI/Together models
    return len(encoding.encode(text))

import re

def clean_response(response_text):
    """Removes unwanted meta-text from AI responses."""
    # Define patterns to remove
    patterns = [
        r"User Satisfaction:.*",
        r"System Improvement Suggestions:.*",
    ]
    
    # Apply regex substitutions
    for pattern in patterns:
        response_text = re.sub(pattern, "", response_text, flags=re.MULTILINE)

    return response_text.strip()

def query_together(query, context="", task_type="default"):
    """Sends query to Together.AI's Mixtral-8x7B-Instruct model with dynamic temperature control."""
    api_key = load_api_key()
    client = Together(api_key=api_key)

    # Task-specific system prompts
    task_prompts = {
        "default": "You are an AI assistant. Your goal is to provide factual, well-structured, and logically sound responses. Do not invent information. If you lack knowledge, say so.",
        "technical": "You are an AI expert in programming, troubleshooting, and explaining technical concepts with accuracy. If uncertain, suggest sources.",
        "research": "You are an AI researcher. Your responses should be fact-based, avoiding speculation. Provide precise, verifiable answers.",
        "creative": "You are an AI storyteller, generating narratives and ideas. If a story is requested, ensure it is structured and engaging.",
        "casual": "You are a conversational AI, engaging in friendly discussions while keeping responses informative and logical."
    }

    # Use the selected prompt or default
    system_prompt = task_prompts.get(task_type, task_prompts["default"])

    # Dynamic temperature adjustment
    task_temperatures = {
        "default": 0.4,
        "technical": 0.3,
        "research": 0.4,
        "creative": 0.8,  # 🔹 Higher randomness for creative writing
        "casual": 0.6
    }

    temperature = task_temperatures.get(task_type, 0.4)

    # Construct the AI prompt
    prompt = f"""
    {system_prompt}

    Rules:
    - If context is provided, ground responses in that context.
    - Do not fabricate details. If unsure, indicate uncertainty.
    - Keep responses clear, logical, and concise.
    
    Previous Conversation:
    {context}

    User Query:
    {query}

    AI Response:
    """

    response = client.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt=prompt,
        temperature=temperature,  # 🔹 Now dynamically controlled
        max_tokens=600,
        top_p=0.9
    )

    # Extract AI response text, clean it, and count tokens
    response_text = clean_response(response.choices[0].text.strip())
    token_count = count_tokens(response_text)

    return response_text, token_count
    
    # Extract AI response text and count tokens
    response_text = response.choices[0].text.strip()
    token_count = count_tokens(response_text)

    return response_text, token_count

if __name__ == "__main__":
    # Test AI Querying
    print("🔹 Running AI Generation Test...")
    response = query_together("What is Weavr AI?")
    print(f"✅ AI Response:\n{response}")
