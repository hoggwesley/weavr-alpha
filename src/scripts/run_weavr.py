import sys
import os
import argparse
import traceback
import asyncio

# Ensure Python finds 'modules/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from modules.config_loader import (
    load_api_key, get_model_name, get_model_api_name, load_config,
    set_model_name, save_config
)
from modules.generation import query_together, handle_command
from modules.structured_memory import initialize_structured_memory, get_structured_memory
from modules.app_state import state

# Argument Parser for CLI
parser = argparse.ArgumentParser(description="Weavr AI - Interactive AI Assistant with Structured Knowledge")
parser.add_argument("--knowledge", action="store_true", help="Enable Knowledge Base")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--knowdir", type=str, help="Specify knowledge base directory")
parser.add_argument("--filelimit", type=int, default=None, help="Maximum number of files to process (default: 100)")
args = parser.parse_args()

# Set debug mode from arguments
state.debug_mode = args.debug

# Initialize knowledge system if requested
if args.knowledge:
    try:
        config = load_config()
        knowledge_base_dir = args.knowdir or config.get("knowledge_base", {}).get("directory", None)
        if not knowledge_base_dir:
            raise ValueError("Knowledge base directory is not specified.")
        if not os.path.isdir(knowledge_base_dir):
            raise ValueError(f"Knowledge base directory does not exist: {knowledge_base_dir}")
        
        # Get file limit from command line args, config, or use default
        file_limit = args.filelimit if args.filelimit is not None else config.get("knowledge_base", {}).get("file_limit", 100)
        
        # Update config with the file limit
        if "knowledge_base" not in config:
            config["knowledge_base"] = {}
        config["knowledge_base"]["file_limit"] = file_limit
        save_config(config)
        
        print(f"📚 Initializing structured knowledge system from: {knowledge_base_dir} (limit: {file_limit} files)")
        state.structured_mem = initialize_structured_memory(knowledge_base_dir, file_limit=file_limit)
        if state.structured_mem:
            state.use_knowledge = True
            print("✅ Structured memory initialized successfully")
        else:
            print("❌ Failed to initialize structured memory")
    except Exception as e:
        print(f"❌ Error initializing knowledge system: {str(e)}")
        if state.debug_mode:
            traceback.print_exc()

# Show startup info
print(f"\n🕷️ Weavr AI is running with {'structured knowledge ✅' if state.use_knowledge else 'no knowledge system ❌'}")
print(f"🔹 Current AI Model: {get_model_name()} ({get_model_api_name()})")

print("\nType your message to chat with AI")
print("Available commands:")
print(" - /knowledge         : Toggle knowledge system on/off")
print(" - /knowledge show    : Show current directory")
print(" - /knowledge status  : Show knowledge base structure")
print(" - /knowledge set     : Set knowledge base directory")
print(" - /knowledge limit   : Set maximum files to process")
print(" - /its              : Iterative Thought Sculpting menu")
print(" - /cot              : Toggle Chain-of-Thought mode (Note: Currently broken, will be reimplemented)")
print(" - /exit             : Quit")

# Main interaction loop
async def main_loop():
    while True:
        try:
            print("\n🎭 --- User --- 🎭")
            query = input("> ").strip()

            if query.lower() == "/exit":
                print("Exiting Weavr AI...")
                break
                
            # Handle slash commands
            if query.startswith('/'):
                # Split command and args, preserving quoted strings
                parts = query.split()
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                result = handle_command(command, args)
                print(result)
                continue
                
            # Handle menu input (1-3)
            if result and "Enter your choice (1-3)" in result:
                if query.isdigit() and 1 <= int(query) <= 3:
                    result = handle_command("/its", [query])
                    print(result)
                    continue
                
            # Handle normal chat interaction
            if state.use_knowledge and state.structured_mem:
                state.structured_mem.check_for_updates()
                
            response = await query_together(query)
            print("\n🤖 --- AI Response --- 🤖\n", response)
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if state.debug_mode:
                traceback.print_exc()

# Run the main loop
asyncio.run(main_loop())

print("Thanks for using Weavr AI!")

# Stop structured memory file watching if running
if state.use_knowledge and state.structured_mem:
    state.structured_mem.stop_file_watching()
