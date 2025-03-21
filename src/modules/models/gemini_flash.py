from google import genai
from google.genai import types
from modules.config_loader import load_api_key, get_system_prompt
import json
import re
from modules.structured_memory import get_structured_memory
from modules.app_state import state

def generate_response(query, context="", conversation_context=""):
    """Handles Gemini AI requests."""
    gemini_api_key = load_api_key()  
    client = genai.Client(api_key=gemini_api_key)
    model_name = "gemini-2.0-flash-001"
    
    # Load system instructions from config
    system_prompt = get_system_prompt()
    
    # Create prompt with context and conversation history
    content_parts = []
    
    # Add conversation context if provided
    if conversation_context.strip():
        content_parts.append(conversation_context)
    
    # Extract potential document titles from the query to improve retrieval
    document_titles = []
    title_pattern = r'"([^"]+)"|\'([^\']+)\''
    for match in re.finditer(title_pattern, query):
        title = match.group(1) or match.group(2)
        if title and len(title) > 3:  # Avoid very short matches
            document_titles.append(title)
    
    # Get relevant structured knowledge instead of RAG context
    try:
        # Only retrieve knowledge if the knowledge system is enabled
        if state.use_knowledge and state.structured_mem:
            structured_memory = get_structured_memory()
            
            # If document titles were found in the query, add a note about them
            if document_titles:
                enhanced_query = query
                for title in document_titles:
                    if not re.search(r'\b' + re.escape(title) + r'\b', enhanced_query, re.IGNORECASE):
                        enhanced_query += f"\n\nNote: Pay special attention to document titled '{title}' if available."
                knowledge_context = structured_memory.get_knowledge_for_query(enhanced_query)
            else:
                knowledge_context = structured_memory.get_knowledge_for_query(query)
            
            if knowledge_context:
                content_parts.append(knowledge_context)
    except Exception as e:
        print(f"❌ Error retrieving structured knowledge: {e}")
        # If structured memory fails, use the provided context as fallback
        pass
    
    # Add the current query with any context (structured memory or fallback)
    if context.strip() and not any("Relevant Knowledge from Your Personal Knowledge Base" in part for part in content_parts):
        content_parts.append(f"Query: {query}\n\nContext: {context}")
    else:
        content_parts.append(f"Query: {query}")
    
    content = "\n\n".join(content_parts)
    
    # Debug the system instruction and content length
    # print(f"DEBUG: System instruction length: {len(system_prompt)}")
    # print(f"DEBUG: Content length (chars): {len(content)}")
    
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
            
            # Calculate token count
            token_count = len(response_text.split())
            return response_text, token_count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0
