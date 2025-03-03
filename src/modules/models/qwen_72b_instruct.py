from together import Together
from modules.config_loader import load_api_key, get_system_prompt  # ✅ Correct import

def generate_response(query, context="", task_type="default"):
    """Handles Qwen-72B AI requests with structured prompt formatting to eliminate hallucinations."""
    api_key = load_api_key()
    client = Together(api_key=api_key)

    model_name = "Qwen/Qwen2-72B-Instruct"

    system_prompt = get_system_prompt()  # ✅ Enforces stricter system prompt usage

    # ✅ Temperature & length control for stability
    temperature = 0.2  # ✅ Lowered to prevent excessive creativity
    if "story" in query.lower() or "explain" in query.lower():
        max_tokens = 1000  # ✅ Give Qwen even more room to complete the final answer
    else:
        max_tokens = min(800, max(200, len(query) * 5))

    # ✅ **Modify prompt for Chain-of-Thought reasoning**
    if task_type == "cot":
        prompt = f"<s> [INST] {system_prompt}\nUser: {query}\nBreak this down step by step. Provide at least 3 to 5 detailed reasoning steps before giving the final answer. Clearly label each step as 'Step 1:', 'Step 2:', etc. Then, provide the final answer in a separate paragraph, ensuring it does not repeat the reasoning steps verbatim. The final answer should summarize the key takeaways. [/INST]"
        prompt = f"""
        <|im_start|>system
        {system_prompt}<|im_end|>

        <|im_start|>user
        {query}\nBreak this down step by step. Provide a structured reasoning process before giving the final answer. Label each step explicitly (Step 1, Step 2, etc.).<|im_end|>

        <|im_start|>assistant
        """

    else:
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

    raw_text = response.choices[0].text.strip()

    # ✅ Extract CoT reasoning steps if enabled
    reasoning_steps = []
    if task_type == "cot":
        reasoning_steps = [step.strip() for step in raw_text.split("\n") if step.strip()]
        
        # ✅ Fix: Only trigger fallback if reasoning is completely empty
        if len(reasoning_steps) > 1:
            final_answer = reasoning_steps.pop()  # Extract final answer
            raw_text = " ".join(reasoning_steps) + "\n\nFinal Answer: " + final_answer  # ✅ Ensure it's fully retrieved

        elif not reasoning_steps:  # If empty, use fallback
            reasoning_steps = ["CoT reasoning not generated—AI needs better prompting."]

    return raw_text, len(raw_text.split()), reasoning_steps
