"""
Core generation module for Weavr AI.
"""
import sys
import os

# ✅ Fix path so 'modules' is recognized correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.config_loader import load_api_key, get_model_name, get_model_api_name, get_system_prompt, load_config, save_config
from modules.cot_engine import MixtralCoTFormatter
from modules.user_instructions import load_instructions, get_combined_prompt
from modules.models import gemini_flash
from modules.context_buffer import context_buffer
from modules.structured_memory import get_structured_memory, initialize_structured_memory
from modules.app_state import state

import importlib
import tiktoken
import re
import traceback

# Enable debugging - commented out for production
DEBUG_MODE = False  # Change to True for debugging

# Add menu context tracking
CURRENT_MENU = "main"

def show_menu():
    """Shows the main menu of available commands."""
    global CURRENT_MENU
    CURRENT_MENU = "main"
    menu = """
🤖 Weavr AI Command Menu
════════════════════════

1. Memory Management
   - View and manage structured knowledge
   - Toggle deep search mode
   - Enable/disable knowledge system
   - Export knowledge

2. Context Management
   - View conversation history
   - Clear conversation memory

3. AI Behavior
   - View/edit instructions
   - Toggle Chain-of-Thought mode

Commands:
- Type a number (1-3) to access a submenu
- Type /menu to show this menu again
- Type /exit to quit
- Or just type your question to chat with AI

What would you like to do? """
    return menu

def query_together(query, context="", task_type="default"):
    """Queries the AI model based on the selected model in config.yaml."""
    try:
        model_name = get_model_name()

        # Get conversation context from the context buffer
        conversation_context = context_buffer.get_formatted_context(query)
        
        # Apply basic relevance filtering to the context buffer
        context_buffer.filter_by_relevance(query)

        # Select the appropriate model based on the configuration
        if model_name == "gemini_flash": # Call the Gemini model
            response = gemini_flash.generate_response(
                query, 
                context=context,  # This is mostly for backward compatibility
                conversation_context=conversation_context
            )
            
            # Store the exchange in the context buffer if response is successful
            if isinstance(response, str):
                context_buffer.add_exchange(query, response)
                return response
            elif isinstance(response, tuple):
                response_text = response[0]
                context_buffer.add_exchange(query, response_text)
                return response_text
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
        else:
            return "Error: Invalid model name in config.yaml"
    except Exception as e:
        return f"Error generating response: {str(e)}"

def handle_command(command, args=None):
    """Handle slash commands."""
    command = command.lower()
    args = args or []
    
    if command == "/knowledge":
        # Handle knowledge subcommands
        if not args:  # No subcommand - toggle knowledge system
            try:
                if not state.use_knowledge:
                    config = load_config()
                    knowledge_base_dir = config.get("knowledge_base", {}).get("directory", None)
                    if not knowledge_base_dir:
                        return "❌ Knowledge base directory not set. Use /knowledge set to configure it."
                    if not os.path.isdir(knowledge_base_dir):
                        return f"❌ Knowledge base directory does not exist: {knowledge_base_dir}"
                    
                    print(f"📚 Initializing knowledge system from: {knowledge_base_dir}")
                    state.structured_mem = initialize_structured_memory(knowledge_base_dir)
                    if state.structured_mem:
                        state.use_knowledge = True
                        return "✅ Knowledge system enabled and initialized successfully."
                    else:
                        return "❌ Failed to initialize knowledge system."
                else:
                    if state.structured_mem:
                        state.structured_mem.stop_file_watching()
                    state.structured_mem = None
                    state.use_knowledge = False
                    return "✅ Knowledge system disabled."
            except Exception as e:
                if state.debug_mode:
                    traceback.print_exc()
                return f"❌ Error toggling knowledge system: {str(e)}"
                
        subcommand = args[0].lower()
        
        if subcommand == "show":  # Show current directory
            config = load_config()
            current_dir = config.get("knowledge_base", {}).get("directory", "Not set")
            return f"📂 Current knowledge base directory: {current_dir}"
            
        elif subcommand == "status":  # Show knowledge base structure
            try:
                config = load_config()
                knowledge_base_dir = config.get("knowledge_base", {}).get("directory", None)
                if not knowledge_base_dir:
                    return "❌ Knowledge base directory not set. Use /knowledge set to configure it."
                if not os.path.isdir(knowledge_base_dir):
                    return f"❌ Knowledge base directory does not exist: {knowledge_base_dir}"
                
                # First show stats from loaded knowledge if available
                output = []
                if state.structured_mem and state.structured_mem.knowledge_store:
                    doc_count = len(state.structured_mem.knowledge_store)
                    section_count = sum(len(doc.get('sections', [])) for doc in state.structured_mem.knowledge_store.values())
                    output.extend([
                        "📊 Knowledge System Status",
                        "════════════════════════",
                        f"- {doc_count} documents loaded",
                        f"- {section_count} total sections indexed",
                        ""
                    ])
                
                # Then show directory structure
                output.extend([f"📚 Knowledge Base Directory ({knowledge_base_dir})", "════════════════════════"])
                
                total_files = 0
                # First collect all paths to sort them
                all_paths = []
                for root, dirs, files in os.walk(knowledge_base_dir):
                    rel_path = os.path.relpath(root, knowledge_base_dir)
                    if rel_path == ".":
                        # Handle files in root directory
                        for f in files:
                            if f.endswith(('.md', '.txt')):
                                all_paths.append((0, False, f))
                                total_files += 1
                        continue
                    
                    # Add directory
                    level = rel_path.count(os.sep)
                    all_paths.append((level, True, os.path.basename(root)))
                    
                    # Add its files
                    for f in files:
                        if f.endswith(('.md', '.txt')):
                            all_paths.append((level + 1, False, f))
                            total_files += 1
                
                # Sort paths by level and name
                all_paths.sort(key=lambda x: (x[0], not x[1], x[2].lower()))
                
                # Generate output
                for level, is_dir, name in all_paths:
                    indent = "  " * level
                    if is_dir:
                        output.append(f"{indent}📂 {name}/")
                    else:
                        output.append(f"{indent}📄 {name}")
                
                if total_files == 0:
                    output.append("\n⚠️  No markdown or text files found in directory.")
                else:
                    output.append(f"\n📊 Total files: {total_files}")
                
                return "\n".join(output)
                    
            except Exception as e:
                if state.debug_mode:
                    traceback.print_exc()
                return f"❌ Error displaying knowledge base status: {str(e)}"
                
        elif subcommand == "set":  # Set directory interactively
            print("Enter the full path to your knowledge base directory:")
            try:
                new_dir = input("> ").strip()
                if not new_dir:
                    return "❌ Operation cancelled."
                
                # Convert to absolute path and normalize
                new_dir = os.path.abspath(os.path.normpath(new_dir))
                
                if not os.path.isdir(new_dir):
                    return f"❌ Directory does not exist: {new_dir}"
                
                # First stop any existing file watching
                if state.structured_mem:
                    state.structured_mem.stop_file_watching()
                    state.structured_mem = None
                    state.use_knowledge = False
                
                # Save the new directory to config
                config = load_config()
                if "knowledge_base" not in config:
                    config["knowledge_base"] = {}
                config["knowledge_base"]["directory"] = new_dir
                
                if not save_config(config):
                    return "❌ Failed to save configuration."
                
                # Initialize with new directory
                print(f"📚 Initializing knowledge system from: {new_dir}")
                state.structured_mem = initialize_structured_memory(new_dir)
                if state.structured_mem:
                    state.use_knowledge = True
                    return f"""✅ Knowledge base directory set and initialized successfully.
📂 Directory: {new_dir}
📊 Documents loaded: {len(state.structured_mem.knowledge_store)}
Use /knowledge status to see full details."""
                else:
                    return "❌ Directory set but failed to initialize knowledge system."
                    
            except Exception as e:
                if state.debug_mode:
                    traceback.print_exc()
                return f"❌ Error setting knowledge directory: {str(e)}"
        
        # If subcommand not recognized, show help
        return """📚 Knowledge System Commands
════════════════════════
/knowledge         - Toggle knowledge system on/off
/knowledge show    - Show current directory
/knowledge status  - Show knowledge base structure
/knowledge set     - Set knowledge base directory"""
            
    elif command == "/cot":
        state.cot_enabled = not state.cot_enabled
        return f"✅ Chain-of-Thought mode {'enabled' if state.cot_enabled else 'disabled'}."
        
    elif command == "/exit":
        return "Exiting..."
        
    return f"❌ Unknown command: {command}"

if __name__ == "__main__":
    print("🔹 Running AI Generation Test...")
    response, token_count, reasoning_steps = query_together("What is Weavr AI?", task_type="cot")
    print(f"✅ AI Response:\n{response}")

    if reasoning_steps:
        print("\n🔍 Chain-of-Thought Reasoning:")
        for step in reasoning_steps:
            print(f" - {step}")
