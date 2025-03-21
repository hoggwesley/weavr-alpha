import yaml

def load_api_key():
    """Loads the Together.AI API key from config.yaml."""
    config_path = r"C:\Users\hoggw\Documents\weavr-alpha\config.yaml"  # Ensure this path is correct
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"ERROR: Config file not found at {config_path}. Please check the path.")
    except yaml.YAMLError as e:
        raise ValueError(f"ERROR: Failed to parse YAML in config.yaml: {e}")

    # Retrieve API key from config
    api_key = config.get('together', {}).get('api_key')

    # Validate API key presence
    if not api_key:
        raise ValueError("ERROR: TOGETHER_API_KEY is missing or empty in config.yaml!")

    return api_key
