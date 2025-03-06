import threading
from modules.file_change_handler import start_observer, stop_observer

# ... existing code ...

observer = None  # Define observer in the global scope

# Define knowledge_base_dir in the global scope
knowledge_base_dir = None

def handle_rag_command(args):
    global rag_mode, observer  # Use the global keyword to modify the global variables
    if args and args[0].lower() == "off":
        if rag_mode:
            print("Turning RAG mode OFF.")
            rag_mode = False
            if observer:
                stop_observer(observer)
                observer = None
            print("✅ RAG Mode DISABLED")
        else:
            print("RAG mode is already OFF.")
    else:
        if not rag_mode:
            print("Turning RAG mode ON.")
            rag_mode = True
            if knowledge_base_dir:
                reindex_flag = threading.Event()
                observer = start_observer(knowledge_base_dir, reindex_flag)
                print("🔄 RAG Mode ENABLED")
            else:
                print("Error: Knowledge base directory not set.")
        else:
            print("RAG mode is already ON.")

# ... existing code ...

def update_instructions():
    global llm
    print("\n--- Current Instructions ---")
    print(llm.get_template())
    print("\n--- Options ---")
    print("1: Keep current instructions")
    print("2: Enter new instructions")
    print("3: Load instructions from file")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        print("Keeping current instructions.")
        return  # Do nothing, keep the current instructions
    elif choice == "2":
        new_instructions = input("Enter the new instructions: ")
        llm.update_template(new_instructions)
        print("Instructions updated successfully!")
    elif choice == "3":
        file_path = input("Enter the path to the instructions file: ")
        try:
            with open(file_path, "r") as file:
                new_instructions = file.read()
            llm.update_template(new_instructions)
            print("Instructions updated successfully!")
        except FileNotFoundError:
            print("Error: File not found.")
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("Invalid choice. Instructions not updated.")

# ... existing code ...

# Updated dialogue and narrative structure

# Function to handle user input and AI response

def handle_input(user_input):
    if user_input.lower() == "who are you?":
        return "I am a reflection of your thoughts and fears, a guide through this maze of uncertainty. But who I am is for you to ponder."
    elif user_input.lower() == "i'm just me, my name is wes":
        return "Ah, Wes. A name, a label, a mere sound. But who are you beyond that? Your name is not your essence, but a tool for communication. You are on a journey, seeking understanding and confronting the unknown."
    elif user_input.lower() == "but i don't know how i got here.":
        return "The path you took, the choices you made, led you here. What matters is that you are here now, facing this strange place and the questions it raises."
    elif user_input.lower() == "i'm not sure honestly":
        return "Not knowing allows for exploration and discovery. Embrace the uncertainty and let it guide you."
    elif user_input.lower() == "i don't feel uneasy, i just feel confused.":
        return "Confusion is a part of the journey. It is a sign that your mind is expanding. Embrace the confusion and let it guide you towards understanding."
    elif user_input.lower() == "is this a test?":
        return "Every moment is a test, shaping who you are and who you will become. This is a journey of self-discovery, not a traditional test."
    elif user_input.lower() == "ok, what do i do? teach me.":
        return "Every moment presents an opportunity for learning. Trust in this process and let it unfold naturally."
    else:
        return "I mean that every moment presents an opportunity for learning, Wes. Every experience, no matter how seemingly mundane, contains within it the potential for growth and understanding."

# ... existing code ...
