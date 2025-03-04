import sys
import os

# ✅ Fix path so 'modules' is recognized correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.config_loader import load_api_key, get_model_name, get_model_api_name, get_system_prompt  # ✅ Now properly imported

import importlib
import tiktoken
import re
import traceback  # For more detailed error logging

# Debug mode
DEBUG_MODE = True

def debug_print(message):
    """Print debug information when debug mode is enabled"""
    if DEBUG_MODE:
        print(f"🔍 DEBUG [generation]: {message}")

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

def validate_cot_structure(response):
    """Ensures CoT output follows a structured format, while allowing slight variations."""
    try:
        debug_print(f"Validating CoT structure for response of length {len(response)}")
        
        # First, clean up the response text to handle potential formatting issues
        response = response.strip()
        
        # Look for any step markers using a more flexible pattern that includes content
        step_pattern = re.compile(r'Step\s*(\d+)\s*:\s*([^\n]*(?:\n(?!Step\s*\d+\s*:)[^\n]*)*)', re.IGNORECASE | re.MULTILINE)
        raw_steps = step_pattern.finditer(response)
        steps = [(m.group(1), m.group(2).strip()) for m in raw_steps]
        debug_print(f"Found {len(steps)} complete steps")
        
        # Validate step content and sequence
        if steps:
            try:
                step_nums = [int(num) for num, _ in steps]
                steps_sequential = sorted(step_nums) == list(range(1, len(step_nums) + 1))
                has_content = all(content.strip() for _, content in steps)
                debug_print(f"Steps sequential: {steps_sequential}, all have content: {has_content}")
                
                # Look for any final answer marker
                answer_markers = [
                    "=== ANSWER ===",
                    "Final answer:",
                    "Final Answer:",
                    "[Final conclusion]:",
                    "In conclusion,",
                    "To conclude,",
                    "In summary,"
                ]
                has_answer = any(marker.lower() in response.lower() for marker in answer_markers)
                debug_print(f"Has answer marker: {has_answer}")
                
                if not has_answer:
                    # Check if there's substantial text after the last step
                    _, last_content = steps[-1]
                    remainder = response.split(last_content, 1)[-1].strip()
                    has_answer = len(remainder) >= 50
                    debug_print(f"Found implied answer: {has_answer}")
                
                return steps_sequential and has_content and (has_answer or len(steps) >= 3)
                
            except ValueError:
                debug_print("Error processing step numbers")
                return False
        
        debug_print("No valid steps found")
        return False
        
    except Exception as e:
        debug_print(f"Error in validate_cot_structure: {str(e)}")
        debug_print(traceback.format_exc())
        return False

def query_together(query, context="", task_type="default"):
    """Routes AI requests while ensuring clean response formatting."""
    try:
        debug_print(f"query_together called with task_type={task_type}")
        
        model_key = get_model_name()  
        model_name = get_model_api_name()

        if not model_name:
            raise ValueError(f"ERROR: Model '{model_key}' is not found in config.yaml!")

        module_name = f"modules.models.{model_key}"  # ✅ Correct path
        debug_print(f"Loading model module: {module_name}")

        try:
            model_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise ImportError(f"ERROR: Model module '{module_name}' not found.")

        if not hasattr(model_module, "generate_response"):
            raise AttributeError(f"ERROR: Module '{module_name}' is missing the 'generate_response' function!")

        # ✅ Enforce system prompt for stricter behavior
        system_prompt = get_system_prompt()

        # ✅ Get AI response (with optional CoT reasoning)
        cot_instruction = ""
        if task_type == "cot":
            debug_print("Preparing CoT instruction based on query")
            if any(word in query.lower() for word in ["why", "how", "explain"]):
                cot_instruction = "Break this down logically, labeling each step explicitly (Step 1, Step 2, etc.)."
            elif "story" in query.lower() or "describe" in query.lower():
                cot_instruction = "Provide a structured narrative to explore this concept."
            elif context and context.strip() != "No relevant retrieval data available.":
                cot_instruction = f"Use the retrieved context below to guide your reasoning. Break your explanation into clear steps."
            else:
                cot_instruction = "Analyze this question in a structured but conversational way."
                
            debug_print(f"CoT instruction: {cot_instruction}")

        context = context if context else "No relevant retrieval data available."
        
        # IMPORTANT FIX: Pass the cot_instruction parameter to the model's generate_response function
        debug_print("Calling model's generate_response function")
        response_text, token_count, reasoning_steps = model_module.generate_response(
            query, 
            context, 
            task_type,
            cot_instruction  # Pass this parameter explicitly
        )
        
        debug_print(f"Got response, token_count={token_count}, reasoning_steps={len(reasoning_steps) if reasoning_steps else 0}")

        # If CoT is enabled, validate response format
        if task_type == "cot":
            debug_print("Validating initial response")
            valid_format = validate_cot_structure(response_text)
            debug_print(f"Initial validation result: {valid_format}")
            
            if not valid_format:
                print("❌ CoT validation failed. Adjusting format and retrying once...")
                debug_print("Retrying with explicit formatting instructions")

                enhanced_cot = """You MUST use this EXACT format:
Step 1: First point of analysis
Step 2: Second point of analysis
Step 3: Further development
Step 4: Final consideration

=== ANSWER ===
[Your final conclusion]"""
                
                retry_response, retry_token_count, retry_reasoning_steps = model_module.generate_response(
                    query,
                    context, 
                    task_type,
                    enhanced_cot
                )
                
                retry_valid = validate_cot_structure(retry_response)
                debug_print(f"Retry validation result: {retry_valid}")

                if retry_valid:
                    debug_print("Using retry response")
                    response_text = retry_response
                    token_count = retry_token_count
                    reasoning_steps = retry_reasoning_steps
                else:
                    print("⚠️ Second CoT validation failed. Using first response.")
                    debug_print("Keeping original response")
            
            # Extract steps and final answer
            step_pattern = re.compile(r'Step\s*(\d+)\s*:\s*([^\n]*(?:\n(?!Step\s*\d+\s*:)[^\n]*)*)', re.IGNORECASE | re.MULTILINE)
            steps = [(m.group(1), m.group(2).strip()) for m in step_pattern.finditer(response_text)]
            reasoning_steps = [f"Step {num}: {content}" for num, content in steps]
            
            # Find final answer with various markers
            answer_pattern = re.compile(
                r'(?:=== ANSWER ===|Final [Aa]nswer:|In conclusion,|To conclude,|In summary,)\s*(.*?)$', 
                re.DOTALL | re.IGNORECASE
            )
            answer_match = answer_pattern.search(response_text)
            
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                # Use text after last step
                final_text = response_text.split(reasoning_steps[-1])[-1].strip() if reasoning_steps else ""
                if len(final_text) >= 50:
                    final_answer = final_text
                else:
                    # Construct from last step
                    _, last_content = steps[-1]
                    final_answer = f"Based on this analysis, {last_content.lower()}"
            
            # Format final response
            formatted_steps = []
            for num, content in steps:
                formatted_steps.append(f"{num}. {content}")
            
            # Clean up final answer
            final_answer = final_answer.strip()
            if not any(final_answer.lower().startswith(start) for start in ["based on", "therefore", "in conclusion", "to conclude", "in summary"]):
                final_answer = "Based on this analysis, " + final_answer[0].lower() + final_answer[1:]
            
            return final_answer, token_count, formatted_steps
            
        else:
            return response_text, token_count, []
        
    except Exception as e:
        debug_print(f"Error in query_together: {str(e)}")
        debug_print(traceback.format_exc())
        return f"Error generating response: {str(e)}", 0, []

if __name__ == "__main__":
    print("🔹 Running AI Generation Test...")
    response, token_count, reasoning_steps = query_together("What is Weavr AI?", task_type="cot")
    print(f"✅ AI Response:\n{response}")

    if reasoning_steps:
        print("\n🔍 Chain-of-Thought Reasoning:")
        for step in reasoning_steps:
            print(f" - {step}")
