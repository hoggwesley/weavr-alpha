from together import Together
from modules.config_loader import load_api_key, get_system_prompt  # ✅ Correct import

def generate_response(query, context="", task_type="default"):
    """Handles Mixtral-8x7B AI requests with proper instruction formatting."""
    api_key = load_api_key()
    client = Together(api_key=api_key)

    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    system_prompt = get_system_prompt()  # ✅ Enforces stricter system prompt usage

    temperature = 0.3  # ✅ Lowered for factual accuracy
    max_tokens = min(400, max(100, len(query) * 5))  # ✅ More dynamic length control

    # **Use Mixtral's recommended [INST] formatting**
    prompt = f"<s> [INST] {system_prompt}\nUser: {query} [/INST]"

    response = client.completions.create(
        model=model_name,
        prompt=prompt.strip(),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.7  # ✅ Tighter control on randomness
    )

    return response.choices[0].text.strip(), len(response.choices[0].text.split())

