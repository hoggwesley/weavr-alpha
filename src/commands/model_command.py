from modules.config_loader import load_config, get_model_name, set_model_name

def execute():
    """Executes the /model command to switch AI models."""
    config = load_config()
    models = list(config.get("together_ai", {}).get("models", {}).items())

    current_index = next((i for i, (key, _) in enumerate(models) if key == get_model_name()), -1)
    new_index = (current_index + 1) % len(models)
    new_model_key = models[new_index][0]
    set_model_name(new_model_key)

    print(f"✅ Model switched to {models[new_index][1]} ({new_model_key})")
