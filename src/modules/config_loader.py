import os
import yaml

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.yaml"))

def load_config():
    """Loads the entire config.yaml file."""
    try:
        with open(CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"ERROR: Config file not found at {CONFIG_PATH}. Please check the path.")
    except yaml.YAMLError as e:
        raise ValueError(f"ERROR: Failed to parse YAML in config.yaml: {e}")

def load_api_key():
    """Retrieves the Together.AI API key from config.yaml."""
    config = load_config()
    api_key = config.get('together_ai', {}).get('api_key')

    if not api_key:
        raise ValueError("ERROR: TOGETHER_API_KEY is missing or empty in config.yaml!")

    return api_key

def get_model_name():
    """Retrieves the currently selected model key from config.yaml."""
    config = load_config()
    models = config.get('together_ai', {}).get('models', {})
    default_model = config.get('together_ai', {}).get('default_model', None)

    if default_model not in models:
        raise ValueError(f"ERROR: Model key '{default_model}' is not found in config.yaml! Ensure it matches a valid key in 'models'.")

    return default_model  # ✅ This now returns the correct key like 'mixtral_8x7b_v01'

def get_model_api_name():
    """Retrieves the full Together AI model name for API calls."""
    config = load_config()
    models = config.get('together_ai', {}).get('models', {})
    model_key = get_model_name()

    return models.get(model_key, None)  # ✅ This now returns the full API model name

def set_model_name(new_model_key):
    """Updates the config.yaml file to switch the default model."""
    config = load_config()

    if new_model_key not in config.get('together_ai', {}).get('models', {}):
        raise ValueError(f"ERROR: Model '{new_model_key}' is not listed in config.yaml!")

    config['together_ai']['default_model'] = new_model_key  # ✅ Update model selection

    # ✅ Save the updated configuration
    with open(CONFIG_PATH, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)

    print(f"✅ Model successfully switched to {new_model_key}")
