from together import Together
from modules.config_loader import load_api_key, get_system_prompt  # ✅ Correct import

def generate_response(query, context="", task_type="default"):
    """Handles Mixtral-8x7B AI requests with proper instruction formatting."""
    api_key = load_api_key()
    client = Together(api_key=api_key)

    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    system_prompt = get_system_prompt()  # ✅ Enforces stricter system prompt usage

    temperature = 0.3  # ✅ Lowered for factual accuracy
    max_tokens = min(1500, max(600, len(query) * 6))  # ✅ Increased range for full CoT + final answer

    # ✅ Modify prompt for Chain-of-Thought reasoning
    if task_type == "cot":
        prompt = f"""
<s> [INST] {system_prompt}
User: {query}

### Step-by-Step Reasoning
Break this problem down step by step before arriving at a conclusion.

- **Step 1:** Identify the core challenges and constraints.
- **Step 2:** Analyze the urgency and dependencies between the issues.
- **Step 3:** Explore possible solutions and evaluate trade-offs.
- **Step 4:** Consider long-term implications and synergies.

### Summary of Key Insights
(Required) Extract the **three most important insights** from the reasoning above. These should be 1-2 sentences each.

### Final Answer
(Required) Using the insights from "Summary of Key Insights," provide **a one-paragraph final answer**. **Do not repeat the reasoning steps.** This should be a recommendation, not a restatement.

[/INST]
""".strip()

    else:
        prompt = f"<s> [INST] {system_prompt}\nUser: {query} [/INST]"

    response = client.completions.create(
        model=model_name,
        prompt=prompt.strip(),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.7  # ✅ Tighter control on randomness
    )

    raw_text = response.choices[0].text.strip()

    # ✅ Extract CoT reasoning steps if enabled
    reasoning_steps = []
    final_answer = ""

    if "### Summary of Key Insights" in raw_text and "### Final Answer" in raw_text:
        parts = raw_text.split("### Summary of Key Insights", 1)[1].split("### Final Answer", 1)
        summary_insights = parts[0].strip()
        final_answer = parts[1].strip()
        
        reasoning_steps = summary_insights.split("\n")

    elif "### Final Answer" in raw_text:
        final_answer = raw_text.split("### Final Answer", 1)[1].strip()
        reasoning_steps = ["Summary missing, but final answer extracted."]

    else:
        reasoning_steps = ["CoT reasoning not properly formatted."]
        final_answer = raw_text  # Fallback to raw response

    return "\n".join(reasoning_steps) + "\n\n**Final Answer:** " + final_answer, len(raw_text.split()), reasoning_steps
