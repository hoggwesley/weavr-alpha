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
from commands.its_command import execute as execute_its_command

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

async def query_together(query, context="", task_type="default"):
    """Queries the AI model based on the selected model in config.yaml."""
    try:
        model_name = get_model_name()

        # Get conversation context from the context buffer
        conversation_context = context_buffer.get_formatted_context(query)
        
        # Apply basic relevance filtering to the context buffer
        context_buffer.filter_by_relevance(query)

        # If ITS is enabled, use the ITS processor
        if state.its_enabled and state.its_processor:
            try:
                # Get knowledge base if available
                knowledge_base = state.structured_mem.knowledge_store if state.use_knowledge and state.structured_mem else None
                
                # Convert conversation context to list of dicts
                chat_history = [
                    {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
                    for i, msg in enumerate(conversation_context.split("\n\n"))
                    if msg.strip()
                ]
                
                # Process using ITS
                response = await state.its_processor.process(query, knowledge_base, chat_history)
                
                # Store the exchange in the context buffer if response is successful
                if response:
                    context_buffer.add_exchange(query, response)
                    return response
                else:
                    raise ValueError("ITS processor returned empty response")
            except Exception as e:
                return f"Error in ITS processing: {str(e)}"

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
    
    # Handle ITS commands
    if command == "/its":
        try:
            # If no args, show menu. Otherwise pass the submenu choice and args
            if not args:
                return execute_its_command()
            else:
                submenu_choice = args[0]
                remaining_args = args[1:] if len(args) > 1 else None
                return execute_its_command(submenu_choice, remaining_args)
        except Exception as e:
            if state.debug_mode:
                traceback.print_exc()
            return f"❌ Error handling ITS command: {str(e)}"
    
    # Handle knowledge subcommands
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
                    
                    # Get file limit from config or use default
                    file_limit = config.get("knowledge_base", {}).get("file_limit", 100)
                    
                    print(f"📚 Initializing knowledge system from: {knowledge_base_dir} (limit: {file_limit} files)")
                    state.structured_mem = initialize_structured_memory(knowledge_base_dir, file_limit=file_limit)
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
                    file_limit = state.structured_mem.file_limit
                    total_eligible_files = state.structured_mem.total_eligible_files
                    excluded_files = state.structured_mem.excluded_files
                    
                    output.extend([
                        "📊 Knowledge System Status",
                        "════════════════════════",
                        f"- {doc_count} documents loaded",
                        f"- {section_count} total sections indexed",
                    ])
                    
                    if total_eligible_files > file_limit:
                        output.append(f"- File limit: {file_limit} (out of {total_eligible_files} eligible files)")
                        output.append(f"- {total_eligible_files - doc_count} files were not processed due to the file limit")
                        
                        # Show excluded files if there are any
                        if excluded_files:
                            output.append("")
                            output.append("📋 Excluded Files (oldest modified first):")
                            # Show up to 10 excluded files to avoid overwhelming output
                            for i, file_path in enumerate(excluded_files[:10]):
                                output.append(f"  - {file_path}")
                            if len(excluded_files) > 10:
                                output.append(f"  - ... and {len(excluded_files) - 10} more files")
                    else:
                        output.append(f"- File limit: {file_limit}")
                    
                    output.append("")
                    
                    # Show only loaded files, organized by directory
                    output.append("📚 Loaded Documents")
                    output.append("════════════════════════")
                    
                    # Get all loaded file paths
                    loaded_files = []
                    for doc_id, doc_data in state.structured_mem.knowledge_store.items():
                        if 'file_path' in doc_data:
                            rel_path = os.path.relpath(doc_data['file_path'], knowledge_base_dir).replace('\\', '/')
                            loaded_files.append(rel_path)
                    
                    # Sort loaded files for consistent display
                    loaded_files.sort()
                    
                    # Group files by directory
                    dir_structure = {}
                    for file_path in loaded_files:
                        parts = file_path.split('/')
                        current_dict = dir_structure
                        
                        # Build directory structure
                        for i, part in enumerate(parts[:-1]):
                            if part not in current_dict:
                                current_dict[part] = {}
                            current_dict = current_dict[part]
                        
                        # Add file to its directory
                        if '_files' not in current_dict:
                            current_dict['_files'] = []
                        current_dict['_files'].append(parts[-1])
                    
                    # Function to recursively print directory structure
                    def print_structure(structure, prefix="", level=0):
                        result = []
                        
                        # Print files in current directory
                        if '_files' in structure:
                            for file in sorted(structure['_files']):
                                result.append(f"{prefix}📄 {file}")
                        
                        # Print subdirectories
                        for dir_name, contents in sorted(structure.items()):
                            if dir_name != '_files':
                                result.append(f"{prefix}📂 {dir_name}/")
                                result.extend(print_structure(contents, prefix + "  ", level + 1))
                        
                        return result
                    
                    # Generate directory structure output
                    output.extend(print_structure(dir_structure))
                    
                    output.append(f"\n📊 Total loaded files: {doc_count}")
                    
                    # If there are excluded files, add a note
                    if excluded_files:
                        output.append(f"⚠️ {len(excluded_files)} files were excluded due to the file limit")
                        output.append("Use /knowledge limit to adjust the file limit")
                
                else:
                    # If no knowledge is loaded, just show the directory
                    output.extend([
                        "📊 Knowledge System Status",
                        "════════════════════════",
                        "- No documents loaded",
                        "",
                        f"📚 Knowledge Base Directory: {knowledge_base_dir}",
                        "Use /knowledge to enable the knowledge system"
                    ])
                
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
                
                # Ask for file limit
                print("Enter maximum number of files to process (default: 100, enter 0 for no limit):")
                file_limit_input = input("> ").strip()
                file_limit = 100  # Default
                if file_limit_input:
                    try:
                        file_limit = int(file_limit_input)
                        if file_limit < 0:
                            print("⚠️ Invalid value, using default limit of 100 files")
                            file_limit = 100
                        elif file_limit == 0:
                            print("⚠️ No file limit set - this may cause performance issues with large directories")
                            file_limit = 10000  # Very high number effectively means no limit
                    except ValueError:
                        print("⚠️ Invalid value, using default limit of 100 files")
                
                # First stop any existing file watching
                if state.structured_mem:
                    state.structured_mem.stop_file_watching()
                    
                    # Explicitly clear the knowledge store to ensure no residual data
                    if hasattr(state.structured_mem, 'knowledge_store'):
                        state.structured_mem.knowledge_store = {}
                        
                    state.structured_mem = None
                    state.use_knowledge = False
                
                # Save the new directory to config
                config = load_config()
                if "knowledge_base" not in config:
                    config["knowledge_base"] = {}
                config["knowledge_base"]["directory"] = new_dir
                config["knowledge_base"]["file_limit"] = file_limit
                
                if not save_config(config):
                    return "❌ Failed to save configuration."
                
                # Initialize with new directory
                print(f"📚 Initializing knowledge system from: {new_dir} (limit: {file_limit} files)")
                state.structured_mem = initialize_structured_memory(new_dir, file_limit=file_limit)
                if state.structured_mem:
                    state.use_knowledge = True
                    return f"""✅ Knowledge base directory set and initialized successfully.
📂 Directory: {new_dir}
📊 Documents loaded: {len(state.structured_mem.knowledge_store)}
📄 File limit: {file_limit}
Use /knowledge status to see full details."""
                else:
                    return "❌ Directory set but failed to initialize knowledge system."
                    
            except Exception as e:
                if state.debug_mode:
                    traceback.print_exc()
                return f"❌ Error setting knowledge directory: {str(e)}"
        
        elif subcommand == "limit":  # Set file limit
            try:
                if len(args) > 1 and args[1].isdigit():
                    new_limit = int(args[1])
                    if new_limit <= 0:
                        return "❌ File limit must be greater than 0."
                else:
                    print("Enter maximum number of files to process (current: " + 
                          str(state.structured_mem.file_limit if state.structured_mem else "not set") + 
                          ", enter 0 for no limit):")
                    limit_input = input("> ").strip()
                    if not limit_input:
                        return "❌ Operation cancelled."
                    
                    try:
                        new_limit = int(limit_input)
                        if new_limit < 0:
                            return "❌ File limit must be greater than or equal to 0."
                        elif new_limit == 0:
                            print("⚠️ No file limit set - this may cause performance issues with large directories")
                            new_limit = 10000  # Very high number effectively means no limit
                    except ValueError:
                        return "❌ Invalid value. Please enter a number."
                
                # Save the new limit to config
                config = load_config()
                if "knowledge_base" not in config:
                    config["knowledge_base"] = {}
                config["knowledge_base"]["file_limit"] = new_limit
                
                if not save_config(config):
                    return "❌ Failed to save configuration."
                
                # If knowledge system is active, reinitialize with new limit
                if state.use_knowledge and state.structured_mem:
                    knowledge_base_dir = state.structured_mem.knowledge_base_dir
                    state.structured_mem.stop_file_watching()
                    state.structured_mem = None
                    
                    print(f"📚 Reinitializing knowledge system with new limit: {new_limit} files")
                    state.structured_mem = initialize_structured_memory(knowledge_base_dir, file_limit=new_limit)
                    if not state.structured_mem:
                        state.use_knowledge = False
                        return "❌ Failed to reinitialize knowledge system with new limit."
                
                return f"✅ File limit updated to {new_limit} files."
                
            except Exception as e:
                if state.debug_mode:
                    traceback.print_exc()
                return f"❌ Error setting file limit: {str(e)}"
        
        elif subcommand == "clear":  # Clear knowledge context
            try:
                if state.structured_mem:
                    # Clear the knowledge store but keep the system enabled
                    state.structured_mem.clear_knowledge()
                    
                    # Prompt for a new directory
                    print("Would you like to set a new knowledge base directory? (y/n)")
                    response = input("> ").strip().lower()
                    
                    if response == 'y' or response == 'yes':
                        print("Enter the full path to your new knowledge base directory:")
                        new_dir = input("> ").strip()
                        if not new_dir:
                            return "✅ Knowledge context cleared. No new directory set."
                        
                        # Convert to absolute path and normalize
                        new_dir = os.path.abspath(os.path.normpath(new_dir))
                        
                        if not os.path.isdir(new_dir):
                            return f"❌ Directory does not exist: {new_dir}\nKnowledge context was cleared but directory was not changed."
                        
                        # Ask for file limit
                        print("Enter maximum number of files to process (default: 100, enter 0 for no limit):")
                        file_limit_input = input("> ").strip()
                        file_limit = 100  # Default
                        if file_limit_input:
                            try:
                                file_limit = int(file_limit_input)
                                if file_limit < 0:
                                    print("⚠️ Invalid value, using default limit of 100 files")
                                    file_limit = 100
                                elif file_limit == 0:
                                    print("⚠️ No file limit set - this may cause performance issues with large directories")
                                    file_limit = 10000  # Very high number effectively means no limit
                            except ValueError:
                                print("⚠️ Invalid value, using default limit of 100 files")
                        
                        # Save the new directory to config
                        config = load_config()
                        if "knowledge_base" not in config:
                            config["knowledge_base"] = {}
                        config["knowledge_base"]["directory"] = new_dir
                        config["knowledge_base"]["file_limit"] = file_limit
                        
                        if not save_config(config):
                            return "❌ Failed to save configuration. Knowledge context was cleared but directory was not changed."
                        
                        # Initialize with new directory
                        print(f"📚 Initializing knowledge system from: {new_dir} (limit: {file_limit} files)")
                        state.structured_mem = initialize_structured_memory(new_dir, file_limit=file_limit)
                        if state.structured_mem:
                            state.use_knowledge = True
                            return f"""✅ Knowledge context cleared and new directory set successfully.
📂 Directory: {new_dir}
📊 Documents loaded: {len(state.structured_mem.knowledge_store)}
📄 File limit: {file_limit}
Use /knowledge status to see full details."""
                        else:
                            return "❌ Knowledge context cleared but failed to initialize with new directory."
                    else:
                        return "✅ Knowledge context cleared. The system is still enabled but all document context has been removed from memory."
                else:
                    return "❌ Knowledge system is not enabled. Nothing to clear."
            except Exception as e:
                if state.debug_mode:
                    traceback.print_exc()
                return f"❌ Error clearing knowledge context: {str(e)}"
        
        # If subcommand not recognized, show help
        return """📚 Knowledge System Commands
════════════════════════
/knowledge         - Toggle knowledge system on/off
/knowledge show    - Show current directory
/knowledge status  - Show knowledge base structure
/knowledge set     - Set knowledge base directory
/knowledge limit   - Set maximum files to process
/knowledge clear   - Clear knowledge context and optionally set a new directory"""
            
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
