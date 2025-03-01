from together import Together
from modules.config_loader import load_api_key

def generate_response(query, context="", task_type="default"):
    """Handles Qwen-72B AI requests with custom settings."""
    api_key = load_api_key()
    client = Together(api_key=api_key)

    model_name = "Qwen/Qwen2-72B-Instruct"

    system_prompts = {
        "default": "You are an advanced AI, providing deep, insightful responses.",
        "technical": "You are a cutting-edge AI expert in technology and engineering.",
        "research": "You are a rigorous researcher. Provide scientifically accurate and sourced responses.",
        "creative": "You are an eloquent storyteller, weaving immersive narratives.",
        "casual": "You are a friendly AI assistant, engaging in relaxed conversation."
    }

    system_prompt = system_prompts.get(task_type, system_prompts["default"])

    temperature = {
        "default": 0.5,
        "technical": 0.3,
        "research": 0.4,
        "creative": 0.9,
        "casual": 0.7
    }[task_type]

    max_tokens = 800

    prompt = f"""
    {system_prompt}

    Rules:
    - If context is provided, ground responses in that context.
    - Keep answers precise, clear, and contextually relevant.
    
    Previous Conversation:
    {context}

    User Query:
    {query}

    AI Response:
    """

    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.85
    )

    return response.choices[0].text.strip(), len(response.choices[0].text.split())
