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
    """Loads the API key from the config."""
    config = load_config()
    
    # Check if we're using Gemini or Together API
    current_model = get_model_name()
    if current_model == "gemini_flash":
        return config.get("gemini", {}).get("api_key")
    else:
        return config.get("together_ai", {}).get("api_key")

def get_system_prompt():
    """Gets the system prompt from the config."""
    config = load_config()
    return config.get("llm", {}).get("system_prompt", 
           "You are a helpful AI assistant that provides detailed and accurate responses.")

def get_model_name():
    """Gets the current model name from the config."""
    config = load_config()
    return config.get("llm", {}).get("model_name", "mixtral_8x7b_v01")

def get_model_api_name():
    """Gets the API name for the current model."""
    config = load_config()
    model_name = get_model_name()
    
    # Handle Gemini models
    if model_name == "gemini_flash":
        return config.get("gemini", {}).get("model_name", "gemini-2.0-flash-001")
    
    # Handle Together.AI models
    return config.get("together_ai", {}).get("models", {}).get(model_name, {}).get("api_name")

def set_model_name(model_name):
    """Sets the current model name in the config."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if "llm" not in config:
        config["llm"] = {}
    
    config["llm"]["model_name"] = model_name
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
