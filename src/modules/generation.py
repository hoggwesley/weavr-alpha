import sys
import os

# ✅ Fix path so 'modules' is recognized correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.config_loader import load_api_key, get_model_name, get_model_api_name, get_system_prompt
from modules.cot_engine import MixtralCoTFormatter  # Updated import path

import importlib
import tiktoken
import re
import traceback  # For more detailed error logging

# Enable debugging - commented out for production
DEBUG_MODE = False  # Change to True for debugging

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

def query_together(query, context="", task_type="default", cot_mode="default"):
    """Routes AI requests while ensuring clean response formatting."""
    print("✅ query_together CALLED")
    try:
        # debug_print(f"query_together called with task_type={task_type}, cot_mode={cot_mode}")
        
        model_key = get_model_name()  
        model_name = get_model_api_name()

        if not model_name:
            raise ValueError(f"ERROR: Model '{model_key}' is not found in config.yaml!")

        module_name = f"modules.models.{model_key}"  # ✅ Correct path
        # debug_print(f"Loading model module: {module_name}")

        try:
            model_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise ImportError(f"ERROR: Model module '{module_name}' not found.")

        if not hasattr(model_module, "generate_response"):
            raise AttributeError(f"ERROR: Module '{module_name}' is missing the 'generate_response' function!")

        # ✅ Enforce system prompt for stricter behavior
        system_prompt = get_system_prompt()

        # Initialize CoT Engine
        formatter = MixtralCoTFormatter()

        # ✅ Get AI response (with optional CoT reasoning)
        if task_type == "cot":
            # debug_print("Generating CoT reasoning steps")

            # EXPERIMENT: Use Mistral-7B for reasoning steps
            try:
                reasoning_module = importlib.import_module("modules.models.mistral_7b_v01")
                # debug_print("Using Mistral-7B for reasoning steps")
            except ModuleNotFoundError:
                # debug_print("Mistral-7B model not found, falling back to selected model")
                reasoning_module = model_module

            # First API call: Generate reasoning steps with Mistral-7B
            reasoning_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}

You are now the reasoning engine for Weavr AI. Your task is to provide clear analysis points.

When analyzing questions about code, files, or directories:
1. Note the specific files or directories mentioned
2. Consider the structure and organization of the codebase
3. Think about how different components interact
4. Consider the most relevant sections based on the headers

You MUST respond using a numbered list format:
1. First key insight about the topic
2. Second key insight about the topic
3. Third key insight about the topic
4. Fourth key insight about the topic

Be concise, clear, and analytical.
<</SYS>>

Analyze this query step by step: {query}

{context if context else ""}
[/INST]"""

            # debug_print(f"Reasoning prompt: {reasoning_prompt[:200]}...")
            # debug_print("Calling generate_response for reasoning steps using Mistral-7B")
            reasoning_text, reasoning_token_count = reasoning_module.generate_response(reasoning_prompt)
            # debug_print(f"Raw reasoning text: {reasoning_text[:200]}...")

            # Extract reasoning steps
            reasoning_steps, _ = formatter.parse_response(reasoning_text)  # Use formatter to parse steps
            # debug_print(f"Extracted {len(reasoning_steps)} reasoning steps from Mistral-7B")

            # Second API call: Synthesize final answer with Mixtral-8x7B
            synthesis_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}

You are the main response engine for Weavr AI, a precise knowledge assistant. Focus exclusively on information available in the knowledge base provided below, avoiding all outside knowledge.

KNOWLEDGE BASE INFORMATION:
{context}

RULES:
1. ONLY use information from the knowledge base above.
2. NEVER introduce facts not present in the knowledge base.
3. If the knowledge base doesn't have relevant information, say "I don't have specific information about that in my knowledge base."
4. DO NOT mention the knowledge base or cite sources in your answer.
5. Structure your answer clearly and concisely.

Question: {query}
<</SYS>>

The user asked: "{query}"

Please answer based solely on the information available in the knowledge base.
[/INST]"""

            # debug_print(f"Synthesis prompt: {synthesis_prompt[:200]}...")
            # debug_print("Calling generate_response for final answer using Mixtral-8x7B")
            final_answer_text, final_answer_token_count = model_module.generate_response(synthesis_prompt)
            # debug_print(f"Raw final answer from Mixtral-8x7B: {final_answer_text[:200]}...")
            
            # Clean the final answer
            final_answer_text = final_answer_text.strip()
            # debug_print(f"Cleaned final answer: {final_answer_text[:100]}...")

            # Combine token counts
            token_count = reasoning_token_count + final_answer_token_count

            # Format the final response - use newlines between steps for better readability
            formatted_text = "\n\n".join(reasoning_steps)
            
            # Only append final answer if we have content
            if final_answer_text:
                formatted_text += f"\n\n=== ANSWER ===\n{final_answer_text}"
            else:
                # Fallback if final answer is empty
                formatted_text += "\n\n=== ANSWER ===\nBased on the analysis, no definitive conclusion can be reached."

            return formatted_text, token_count, reasoning_steps
        
        else:
            # Non-CoT response
            prompt = f"""<s>[INST] <<SYS>>
{system_prompt}

You are the main response engine for Weavr AI, a precise knowledge assistant. Focus exclusively on information available in the knowledge base provided below, avoiding all outside knowledge.

KNOWLEDGE BASE INFORMATION:
{context}

RULES:
1. ONLY use information from the knowledge base above.
2. NEVER introduce facts not present in the knowledge base.
3. If the knowledge base doesn't have relevant information, say "I don't have specific information about that in my knowledge base."
4. DO NOT mention the knowledge base or cite sources in your answer.
5. Structure your answer clearly and concisely.

Question: {query}
<</SYS>>

The user asked: "{query}"

Please answer based solely on the information available in the knowledge base.
[/INST]"""
            response_text, token_count = model_module.generate_response(prompt)
            return response_text, token_count, []
        
    except Exception as e:
        # debug_print(f"Error in query_together: {str(e)}")
        # debug_print(traceback.format_exc())
        return f"Error generating response: {str(e)}", 0, []

if __name__ == "__main__":
    print("🔹 Running AI Generation Test...")
    response, token_count, reasoning_steps = query_together("What is Weavr AI?", task_type="cot")
    print(f"✅ AI Response:\n{response}")

    if reasoning_steps:
        print("\n🔍 Chain-of-Thought Reasoning:")
        for step in reasoning_steps:
            print(f" - {step}")
