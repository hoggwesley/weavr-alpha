from modules.user_instructions import load_instructions, save_instructions

def execute(submenu_choice=None):
    """Executes the /instructions command to view or edit AI behavior instructions."""
    current_instructions = load_instructions()
    
    if submenu_choice == "1":  # View instructions
        if current_instructions:
            return f"""
📝 Current AI Instructions
═════════════════════════
{current_instructions}

Press Enter to return to AI Behavior menu."""
        else:
            return "No custom instructions set. The AI is using default behavior.\n\nPress Enter to return to AI Behavior menu."
            
    elif submenu_choice == "2":  # Edit instructions
        return """
📝 Edit AI Instructions
═════════════════════════
Enter new instructions below. Press Ctrl+Z then Enter when finished.
To cancel, just press Enter without typing anything.

Current instructions:
""" + (current_instructions if current_instructions else "(none)")
            
    elif submenu_choice == "3":  # Delete instructions
        if save_instructions(""):
            return "✅ Instructions deleted successfully.\n\nPress Enter to return to AI Behavior menu."
        else:
            return "❌ Failed to delete instructions.\n\nPress Enter to return to AI Behavior menu."
            
    # If no submenu choice, treat as saving new instructions
    elif submenu_choice:
        if save_instructions(submenu_choice):
            return "✅ Instructions saved successfully.\n\nPress Enter to return to AI Behavior menu."
        else:
            return "❌ Failed to save instructions.\n\nPress Enter to return to AI Behavior menu."
            
    return """
📝 AI Behavior Instructions
═════════════════════════
1. View current instructions
2. Edit instructions 
3. Delete instructions
4. Back to main menu

Enter your choice (1-4): """
