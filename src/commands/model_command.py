from modules.config_loader import load_config, get_model_name, set_model_name

def execute():
    """Shows a numbered menu of AI models for the user to select from."""
    config = load_config()
    
    # Get models and merge them into one list
    models = []
    
    # Get Together.AI models
    together_models = list(config.get("together_ai", {}).get("models", {}).items())
    
    # Filter out mistral_7b_v01 which is only used for CoT
    for model_key, model_info in together_models:
        if model_key != "mistral_7b_v01" and model_key != "gemini_flash":  # Skip mistral_7b and gemini from together models
            models.append((model_key, model_info))
    
    # Add Gemini model to the list if it exists (ensure it's only added once)
    gemini_config = config.get("gemini", {})
    if gemini_config and "model_name" in gemini_config:
        gemini_model = ("gemini_flash", {"api_name": gemini_config.get("model_name")})
        # Ensure we're not adding duplicate models
        if not any(model[0] == "gemini_flash" for model in models):
            models.append(gemini_model)
    
    if not models:
        print("❌ Error: No models found in the configuration.")
        return

    # Get current model for highlighting
    current_model = get_model_name()
    
    # Display model menu with clean formatting
    print("\n🤖 Available Models:")
    for i, (model_key, model_info) in enumerate(models, 1):
        current_marker = "➡️ " if model_key == current_model else "   "
        api_name = model_info.get('api_name', 'Unknown')
        # Improved formatting for model display
        model_display = f"{current_marker}[{i}] {model_key}"
        # Add padding to align all descriptions
        padding = " " * (25 - len(model_display))
        print(f"{model_display}{padding}: {api_name}")
    
    # Get user selection
    while True:
        selection = input("\nEnter model number to use (or press Enter to cancel): ").strip()
        
        if not selection:
            print("Model selection cancelled.")
            return
        
        try:
            selection_idx = int(selection) - 1
            if 0 <= selection_idx < len(models):
                selected_model = models[selection_idx][0]
                set_model_name(selected_model)
                print(f"✅ Model successfully switched to {selected_model}")
                print(f"✅ Now using: {models[selection_idx][1].get('api_name')}")
                return
            else:
                print("❌ Invalid selection. Please choose a number from the list.")
        except ValueError:
            print("❌ Please enter a valid number.")
