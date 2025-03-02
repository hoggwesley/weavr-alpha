from together import Together
from modules.config_loader import load_api_key, get_system_prompt  # ✅ Correct import

def generate_response(query, context="", task_type="default"):
    """Handles Qwen-72B AI requests with structured prompt formatting to eliminate hallucinations."""
    api_key = load_api_key()
    client = Together(api_key=api_key)

    model_name = "Qwen/Qwen2-72B-Instruct"

    system_prompt = get_system_prompt()  # ✅ Enforces stricter system prompt usage

    # **Temperature & length control for stability**
    temperature = 0.2  # ✅ Lowered to prevent excessive creativity
    # ✅ More flexible token control to avoid premature cut-offs
    if "story" in query.lower() or "explain" in query.lower():
        max_tokens = 400  # Allow longer responses for storytelling & explanations
    else:
        max_tokens = min(300, max(75, len(query) * 4))  # More balanced length control

    # ✅ **Manually format prompt according to Qwen2's chat template**
    prompt = f"""
    <|im_start|>system
    {system_prompt}<|im_end|>
    
    <|im_start|>user
    {query}<|im_end|>
    
    <|im_start|>assistant
    """

    response = client.completions.create(
        model=model_name,
        prompt=prompt.strip(),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.7,
        stop=["<|im_end|>", "\n\n"]  # ✅ Encourage logical stopping points
)

    return response.choices[0].text.strip(), len(response.choices[0].text.split())
