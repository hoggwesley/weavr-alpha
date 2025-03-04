from together import Together
from modules.config_loader import load_api_key, get_system_prompt
import re
import traceback

# Enable debug mode
DEBUG_MODE = True

def debug_print(message):
    """Print debug information when debug mode is enabled"""
    if DEBUG_MODE:
        print(f"🔍 DEBUG [mixtral]: {message}")

def generate_response(query, context="", task_type="default", cot_instruction=None):
    """Handles Mixtral-8x7B AI requests with proper instruction formatting."""
    debug_print(f"generate_response called with task_type={task_type}, cot_instruction={cot_instruction}")
    
    try:
        api_key = load_api_key()
        client = Together(api_key=api_key)

        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        debug_print(f"Using model: {model_name}")

        system_prompt = get_system_prompt()

        temperature = 0.3
        max_tokens = 2048  # Increase max tokens for longer responses

        if task_type == "cot":
            debug_print("Building CoT prompt format")
            cot_guidance = cot_instruction if cot_instruction else "Break down your reasoning into clear, numbered steps."
            debug_print(f"CoT guidance: {cot_guidance}")
            
            # Enhanced prompt with strict formatting guidance
            prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>>

Question: {query}

{cot_guidance}

You MUST follow this EXACT format with no deviations:

Step 1: [First point]
- Detailed explanation
- Supporting details

Step 2: [Second point]
- Further analysis
- Key insights

Step 3: [Third point]
- Additional perspectives
- Important considerations

Step 4: [Final point]
- Synthesis of insights
- Key conclusions

=== ANSWER ===
[Provide a clear, concise final answer that draws from the analysis above]

[/INST]"""

            debug_print(f"CoT prompt prepared, length: {len(prompt)}")
            temperature = 0.65  # Higher temperature for creative reasoning
            
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
        else:
            prompt = f"<s>[INST] {system_prompt}\nUser: {query} [/INST]"
            debug_print("Standard (non-CoT) prompt prepared")
            
            response = client.completions.create(
                model=model_name,
                prompt=prompt.strip(),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.7
            )

        raw_text = response.choices[0].text.strip()
        token_count = len(raw_text.split())
        debug_print(f"Received {token_count} tokens in response")

        if task_type != "cot":
            debug_print("Returning standard response (no CoT processing)")
            return raw_text, token_count, []

        debug_print("Processing CoT response")
        debug_print(f"Raw response first 100 chars: {raw_text[:100]}...")

        # Enhanced step pattern matching
        COT_STEP_PATTERN = re.compile(
            r"Step\s*(\d+)\s*:\s*([^\n]*(?:\n(?!Step\s*\d+\s*:)[^\n]*)*)", 
            re.IGNORECASE | re.MULTILINE
        )
        ANSWER_PATTERN = re.compile(
            r"===\s*ANSWER\s*===\s*(.*?)(?:\[/INST\]|$)",
            re.DOTALL | re.IGNORECASE
        )

        # Extract reasoning steps with improved pattern
        steps = [(m.group(1), m.group(2).strip()) for m in COT_STEP_PATTERN.finditer(raw_text)]
        reasoning_steps = []
        
        debug_print(f"Found {len(steps)} step matches with regex")
        
        if steps:
            for step_num, step_text in steps:
                step_content = f"Step {step_num}: {step_text}"
                reasoning_steps.append(step_content)
                debug_print(f"Added step {step_num}: {step_text[:30]}...")

        # Extract final answer with more robust pattern matching
        match_answer = ANSWER_PATTERN.search(raw_text)
        if match_answer:
            debug_print("Found answer delimiter in response")
            final_answer = match_answer.group(1).strip()
        else:
            debug_print("No standard answer delimiter found, looking for alternative formats")
            # Try alternative answer formats
            alt_patterns = [
                r"(?:Final [Aa]nswer|In conclusion|To conclude|In summary):\s*(.*?)(?:\[/INST\]|$)",
                r"\n\n([^S\n].*?)(?:\[/INST\]|$)"  # Catch non-step text at end
            ]
            
            for pattern in alt_patterns:
                match = re.search(pattern, raw_text, re.DOTALL)
                if match:
                    final_answer = match.group(1).strip()
                    debug_print("Found answer with alternative pattern")
                    break
            else:
                # If no patterns match, use text after last step
                if reasoning_steps:
                    parts = raw_text.split(reasoning_steps[-1], 1)
                    final_answer = parts[1].strip() if len(parts) > 1 else reasoning_steps[-1]
                else:
                    final_answer = raw_text
        
        debug_print(f"Final answer (first 50 chars): {final_answer[:50]}...")
        debug_print(f"Returning {len(reasoning_steps)} reasoning steps")
        
        # Format complete response
        formatted_text = ""
        if reasoning_steps:
            formatted_text = "\n\n".join(reasoning_steps)
        if final_answer:
            formatted_text += f"\n\n=== ANSWER ===\n{final_answer}"
        
        # Ensure final answer starts with a capital letter
        if formatted_text and formatted_text[0].islower():
            formatted_text = formatted_text[0].upper() + formatted_text[1:]
        
        return formatted_text, token_count, reasoning_steps
        
    except Exception as e:
        debug_print(f"Error in generate_response: {str(e)}")
        debug_print(traceback.format_exc())
        return f"An error occurred: {str(e)}", 0, ["Error generating response."]
