from together import Together
from modules.config_loader import load_api_key

def generate_response(query, context="", task_type="default"):
    """Handles Mixtral-8x7B v0.1 AI requests with custom settings."""
    api_key = load_api_key()
    client = Together(api_key=api_key)

    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    system_prompts = {
        "default": "You are a logical AI assistant providing accurate, structured responses.",
        "technical": "You are an AI expert in programming and troubleshooting.",
        "research": "You are an AI researcher. Provide only factual, well-researched information.",
        "creative": "You are a creative storyteller generating engaging narratives.",
        "casual": "You are a conversational AI, keeping discussions friendly and engaging."
    }

    system_prompt = system_prompts.get(task_type, system_prompts["default"])

    temperature = {
        "default": 0.4,
        "technical": 0.3,
        "research": 0.4,
        "creative": 0.8,
        "casual": 0.6
    }[task_type]

    max_tokens = 600

    prompt = f"""
    {system_prompt}

    Rules:
    - If context is provided, integrate it carefully.
    - Do not fabricate details. If unsure, indicate uncertainty.
    - Keep responses logical, structured, and clear.
    
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
        top_p=0.9
    )

    return response.choices[0].text.strip(), len(response.choices[0].text.split())
