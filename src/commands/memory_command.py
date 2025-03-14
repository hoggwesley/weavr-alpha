"""
Command handler for memory management in Weavr AI.
"""
from modules.structured_memory import get_structured_memory
from modules.config_loader import load_config, save_config

def execute(submenu_choice=None, query=None):
    """Handle memory management commands."""
    try:
        config = load_config()
        
        # Special handling for enabling/disabling knowledge system
        if submenu_choice == "3":  # Enable/disable knowledge system
            if "memory" not in config:
                config["memory"] = {}
            current = config["memory"].get("enabled", True)
            config["memory"]["enabled"] = not current
            if save_config(config):
                new_state = "enabled" if not current else "disabled"
                return f"✅ Knowledge system {new_state}.\n\nPress Enter to return to Memory Management menu."
            return "❌ Failed to toggle knowledge system.\n\nPress Enter to return to Memory Management menu."

        # For all other commands, check if structured memory is available
        try:
            structured_mem = get_structured_memory()
        except ValueError:
            if submenu_choice is None:
                # Show menu even if structured memory is not initialized
                return """
📚 Memory Management
════════════════════════
1. Show knowledge stats (unavailable)
2. Toggle deep search mode (unavailable)
3. Enable/disable knowledge system
4. Update knowledge base (unavailable)
5. Export knowledge store (unavailable)
6. Back to main menu

Enter your choice (1-6): """
            else:
                return "❌ This command requires structured memory to be initialized. Enable the knowledge system first.\n\nPress Enter to return to Memory Management menu."
        
        if submenu_choice == "1":  # Show knowledge stats
            doc_count = len(structured_mem.knowledge_store)
            section_count = sum(len(doc.get('sections', [])) for doc in structured_mem.knowledge_store.values())
            deep_search = config.get("memory", {}).get("deep_search_enabled", False)
            return f"""
📚 Knowledge Base Statistics
════════════════════════
- {doc_count} documents loaded
- {section_count} total sections indexed
- Knowledge base is {'up to date' if not structured_mem.needs_update() else 'needs update'}
- Deep search mode is {'enabled' if deep_search else 'disabled'}

Press Enter to return to Memory Management menu."""

        elif submenu_choice == "2":  # Toggle deep search mode
            if "memory" not in config:
                config["memory"] = {}
            current = config["memory"].get("deep_search_enabled", False)
            config["memory"]["deep_search_enabled"] = not current
            if save_config(config):
                new_state = "enabled" if not current else "disabled"
                return f"✅ Deep search mode {new_state}.\n\nPress Enter to return to Memory Management menu."
            return "❌ Failed to toggle deep search mode.\n\nPress Enter to return to Memory Management menu."

        elif submenu_choice == "4":  # Update knowledge base
            structured_mem.update_knowledge()
            return "✅ Structured knowledge has been updated.\n\nPress Enter to return to Memory Management menu."
            
        elif submenu_choice == "5":  # Export knowledge
            result = structured_mem.export_knowledge("yaml")
            return f"""✅ Knowledge store exported:

{result}

Press Enter to return to Memory Management menu."""
            
        # Default menu
        return """
📚 Memory Management
════════════════════════
1. Show knowledge stats
2. Toggle deep search mode
3. Enable/disable knowledge system
4. Update knowledge base
5. Export knowledge store
6. Back to main menu

Enter your choice (1-6): """
            
    except Exception as e:
        return f"❌ Error handling memory command: {str(e)}\n\nPress Enter to return to Memory Management menu."