"""
Command handler for AI behavior settings in Weavr AI.
"""
from modules.config_loader import load_config, save_config

def execute(submenu_choice=None, query=None):
    """Handle AI behavior commands."""
    try:
        config = load_config()
        if "behavior" not in config:
            config["behavior"] = {}

        if submenu_choice == "1":  # View instructions
            instructions = config["behavior"].get("instructions", "No custom instructions set.")
            return f"""
📝 Current AI Instructions
════════════════════════
{instructions}

Press Enter to return to Behavior Settings menu."""

        elif submenu_choice == "2":  # Edit instructions
            if not query:
                return """Enter new instructions for the AI (or 'clear' to remove custom instructions):

Type your instructions: """
            if query.lower() == "clear":
                config["behavior"]["instructions"] = ""
                if save_config(config):
                    return "✅ Custom instructions cleared.\n\nPress Enter to return to Behavior Settings menu."
            else:
                config["behavior"]["instructions"] = query
                if save_config(config):
                    return "✅ Custom instructions updated.\n\nPress Enter to return to Behavior Settings menu."
            return "❌ Failed to update instructions.\n\nPress Enter to return to Behavior Settings menu."

        elif submenu_choice == "3":  # Toggle CoT
            current = config["behavior"].get("cot_enabled", False)
            config["behavior"]["cot_enabled"] = not current
            if save_config(config):
                new_state = "enabled" if not current else "disabled"
                return f"✅ Chain-of-Thought mode {new_state}.\n\nPress Enter to return to Behavior Settings menu."
            return "❌ Failed to toggle Chain-of-Thought mode.\n\nPress Enter to return to Behavior Settings menu."

        # Default menu
        return """
🤖 AI Behavior Settings
════════════════════════
1. View instructions
2. Edit instructions
3. Toggle Chain-of-Thought mode
4. Back to main menu

Enter your choice (1-4): """

    except Exception as e:
        return f"❌ Error handling behavior command: {str(e)}\n\nPress Enter to return to Behavior Settings menu." 