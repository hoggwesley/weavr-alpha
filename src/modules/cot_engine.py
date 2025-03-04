import re
import traceback

from modules.config_loader import get_model_name, get_model_api_name

# Enable debugging
DEBUG_MODE = True

def debug_print(message):
    """Print debug information when debug mode is enabled"""
    if DEBUG_MODE:
        print(f"🔍 DEBUG [cot_engine]: {message}")

class MixtralCoTFormatter:
    """Formatter for Mixtral model's CoT responses"""
    
    def __init__(self):
        # More robust regex pattern
        self.step_pattern = re.compile(
            r"(?:^|\n)(?:Step|step|[0-9]+[\.\):])\s*([^\n]*(?:\n(?!(?:Step|step|[0-9]+[\.\):])\s|===\s*ANSWER\s*===)[^\n]*)*)",
            re.IGNORECASE | re.MULTILINE
        )
        # More robust answer pattern
        self.answer_pattern = re.compile(
            r"\s*===\s*ANSWER\s*===\s*\n(.*?)(?:\[/INST\]|$)",
            re.DOTALL | re.IGNORECASE
        )
        self.alternative_answer_pattern = re.compile(
            r"(?:Final [Aa]nswer:|In conclusion:|Thus,|Therefore,|To summarize:|Overall,)\s*(.*?)$",
            re.DOTALL
        )

    def clean_response(self, response_text):
        """Cleans the raw response before parsing."""
        # Remove leading/trailing whitespace
        cleaned_text = response_text.strip()
        # Remove extra newlines and spaces
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = re.sub(r" +", " ", cleaned_text)
        return cleaned_text

    def format_prompt(self, query, context="", cot_instruction=""):
        """Formats the prompt for Mixtral CoT."""
        system_prompt = "You are a helpful AI assistant that provides detailed and accurate responses."
        
        # Create a strong, explicit CoT instruction
        effective_instruction = """
You MUST respond using the following format:
1. First point of analysis
2. Second point of analysis 
3. Further analysis
4. Additional context
"""
        
        if cot_instruction:
            effective_instruction = cot_instruction
            
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}

{effective_instruction}
<</SYS>>

{query}

{context if context != "No relevant retrieval data available." else ""}
[/INST]"""
        return prompt

    def parse_response(self, response_text):
        """Parses the Mixtral CoT response to extract reasoning steps and the final answer."""
        reasoning_steps = []
        final_answer = ""

        # Clean the response before parsing
        cleaned_response = self.clean_response(response_text)
        debug_print(f"Cleaned response: {cleaned_response[:100]}...")
        
        # Check for numbered list format (1. 2. 3. etc)
        number_pattern = re.compile(r"(?:^|\n)([0-9]+)[\.\):][ \t]*([^\n]*(?:\n(?![0-9]+[\.\):]|===\s*ANSWER\s*===)[^\n]*)*)", re.MULTILINE)
        numbered_steps = number_pattern.findall(cleaned_response)
        
        # If numbered format is found, convert to Step format
        if numbered_steps:
            debug_print(f"Found {len(numbered_steps)} numbered steps")
            for step_num, step_text in numbered_steps:
                # Keep the step number but ensure consistent formatting
                step_content = f"Step {step_num}: {step_text.strip()}"
                reasoning_steps.append(step_content)
                debug_print(f"Added numbered step {step_num}")
        else:
            # Try finding step format
            step_pattern = re.compile(
                r"(?:^|\n)(?:Step|step)\s*([0-9]+)\s*:[ \t]*([^\n]*(?:\n(?!(?:Step|step)\s*[0-9]+|===\s*ANSWER\s*===)[^\n]*)*)", 
                re.IGNORECASE | re.MULTILINE
            )
            steps = [(m.group(1), m.group(2).strip()) for m in step_pattern.finditer(cleaned_response)]
            
            if steps:
                debug_print(f"Found {len(steps)} steps with 'Step X:' format")
                for step_num, step_text in steps:
                    step_content = f"Step {step_num}: {step_text}"
                    reasoning_steps.append(step_content)
                    debug_print(f"Added step {step_num}")

        # Extract answer with different patterns
        if not final_answer:  # Only search if we don't already have an answer
            match_answer = self.answer_pattern.search(cleaned_response)
            if match_answer:
                debug_print("Found answer with '=== ANSWER ===' delimiter")
                final_answer = match_answer.group(1).strip()
            else:
                # Try alternative patterns
                alt_match = self.alternative_answer_pattern.search(cleaned_response)
                if alt_match:
                    debug_print("Found answer with alternative delimiter")
                    final_answer = alt_match.group(1).strip()
                elif reasoning_steps:
                    # If we have steps but no explicit answer, use text after last step
                    debug_print("No answer delimiter found, using text after last step")
                    parts = cleaned_response.split(reasoning_steps[-1].strip(), 1)
                    if len(parts) > 1 and len(parts[1].strip()) > 50:
                        final_answer = parts[1].strip()
                    else:
                        # Construct answer from what we have
                        final_answer = f"Based on the analysis, {parts[1].strip() if len(parts) > 1 else 'no clear conclusion could be reached.'}"
                else:
                    # No steps and no explicit answer - use the whole response
                    debug_print("No steps or answer delimiter found, using entire response")
                    final_answer = cleaned_response

        # If no steps found but answer exists, create dummy step
        if not reasoning_steps and final_answer:
            debug_print("Creating dummy reasoning step from answer")
            reasoning_steps = ["Step 1: Initial analysis - The question requires examining key aspects and relationships."]

        return reasoning_steps, final_answer
