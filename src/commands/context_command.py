"""
Command handler for context management in Weavr AI.
"""
from modules.context_buffer import ContextBuffer

def execute(submenu_choice=None, query=None):
    """Handle context management commands."""
    try:
        context_buffer = ContextBuffer()

        if submenu_choice == "1":  # View history
            history = context_buffer.get_formatted_context()
            if not history:
                return "No conversation history available.\n\nPress Enter to return to Context Management menu."
            return f"""
💬 Conversation History
════════════════════════
{history}

Press Enter to return to Context Management menu."""

        elif submenu_choice == "2":  # Clear history
            context_buffer.clear()
            return "✅ Conversation history cleared.\n\nPress Enter to return to Context Management menu."

        elif submenu_choice == "3":  # Filter history
            if not query:
                return """Enter keywords to filter conversation history:

Type your keywords: """
            filtered = context_buffer.filter_by_relevance(query)
            if not filtered:
                return "No relevant conversation history found.\n\nPress Enter to return to Context Management menu."
            return f"""
💬 Filtered Conversation History
════════════════════════
{filtered}

Press Enter to return to Context Management menu."""

        # Default menu
        return """
💬 Context Management
════════════════════════
1. View conversation history
2. Clear conversation history
3. Filter conversation history
4. Back to main menu

Enter your choice (1-4): """

    except Exception as e:
        return f"❌ Error handling context command: {str(e)}\n\nPress Enter to return to Context Management menu."