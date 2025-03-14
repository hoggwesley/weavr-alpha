import os
import yaml
import json

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.yaml"))

def get_config_path():
    """Get the path to the config file."""
    return CONFIG_PATH

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
    return config.get("gemini", {}).get("api_key")

def get_system_prompt():
    """Gets the system prompt from the config."""
    config = load_config()
    return config.get("llm", {}).get("system_prompt", 
           "You are a helpful AI assistant that provides detailed and accurate responses.")

def get_model_name():
    """Gets the current model name from the config."""
    config = load_config()
    return config.get("llm", {}).get("model_name", "gemini_flash")

def get_model_api_name():
    """Gets the API name for the current model."""
    config = load_config()
    return config.get("gemini", {}).get("model_name", "gemini-2.0-flash-001")

def save_config(config):
    """Saves the entire config dictionary back to config.yaml."""
    try:
        with open(CONFIG_PATH, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        return True
    except Exception as e:
        print(f"ERROR: Failed to save config: {e}")
        return False

def set_model_name(model_name):
    """Sets the current model name in the config."""
    config = load_config()
    
    if "llm" not in config:
        config["llm"] = {}
    
    config["llm"]["model_name"] = model_name
    return save_config(config)

def get_app_state_path():
    """Get the path to the app state file."""
    config_dir = os.path.dirname(CONFIG_PATH)
    return os.path.join(config_dir, 'app_state.json')

def load_app_state():
    """Load the saved application state."""
    state_path = get_app_state_path()
    if not os.path.exists(state_path):
        return {
            "menu_enabled": False,
            "deep_search_enabled": False,
            "chain_of_thought_enabled": False,
            "knowledge_system_enabled": False,
            "last_model": get_model_name()
        }
    
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading app state: {str(e)}")
        return {}

def save_app_state(state):
    """Save the current application state."""
    try:
        state_path = get_app_state_path()
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving app state: {str(e)}")
        return False
