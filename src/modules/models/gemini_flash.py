from google import genai
from google.genai import types
from modules.config_loader import load_api_key, get_system_prompt
import json
import re

def generate_response(query, context=""):
    """Handles Gemini AI requests."""
    gemini_api_key = load_api_key()  
    client = genai.Client(api_key=gemini_api_key)
    model_name = "gemini-2.0-flash-001"
    
    # Load system instructions from config
    system_prompt = get_system_prompt()
    
    # Create prompt with context
    content = f"Query: {query}\n\nContext: {context}" if context.strip() else query
    
    # Debug the system instruction
    print(f"DEBUG: System instruction length: {len(system_prompt)}")
    
    try:
        # Configure the generation parameters with system instruction and safety settings
        gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=1024,
            top_p=0.95,
            top_k=40,
            # Use the most permissive safety settings possible
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE" 
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_ONLY_HIGH"
                )
            ]
        )
        
        # Make the API call with the proper configuration
        response = client.models.generate_content(
            model=model_name,
            contents=content,
            config=gen_config
        )
        
        # Check if response has the expected structure and clean up the formatting
        if hasattr(response, 'text'):
            response_text = response.text.strip()
            
            # Clean up the text
            # Remove "Query:", "Answer:", etc. at the beginning
            response_text = re.sub(r'^(Query|Answer|Response):?\s*', '', response_text)
            
            # Clean up code blocks
            if response_text.startswith('```'):
                response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            return response_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
