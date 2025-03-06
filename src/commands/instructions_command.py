from modules.user_instructions import load_instructions, save_instructions

def execute():
    """Executes the /instructions command to view or edit AI behavior instructions."""
    current_instructions = load_instructions()
    
    print("\n📝 AI Behavior Instructions")
    print("---------------------------")
    if current_instructions:
        print("Current instructions:")
        print(current_instructions)
    else:
        print("No custom instructions set. The AI is using default behavior.")
    
    print("\nOptions:")
    print("1. Edit instructions")
    print("2. Delete instructions")
    print("3. Leave instructions unchanged")
    
    choice = input("\nEnter your choice (1-3, or press Enter to leave as is): ").strip()
    
    if choice == "1":
        # Note: Windows users must press Ctrl+Z then Enter to signal EOF.
        print("\nEnter new instructions (press Ctrl+Z then Enter on a new line when finished):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass  # EOF signaled
        new_instructions = "\n".join(lines)
        if save_instructions(new_instructions):
            print("✅ Instructions saved successfully.")
        else:
            print("❌ Failed to save instructions.")
    
    elif choice == "2":
        if save_instructions(""):
            print("✅ Instructions deleted successfully.")
        else:
            print("❌ Failed to delete instructions.")
    
    elif choice == "3" or choice == "":
        print("Instructions unchanged.")
    
    else:
        print("Invalid choice. Instructions unchanged.")
