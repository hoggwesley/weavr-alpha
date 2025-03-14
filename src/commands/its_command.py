"""
Command handler for Iterative Thought Sculpting (ITS) in Weavr AI.
"""
from modules.reasoning import ITSCommandHandler, ITSConfig, ITSProcessor
from modules.config_loader import load_config, save_config, load_api_key, get_model_api_name
from modules.app_state import state

def execute(submenu_choice=None, args=None):
    """Handle ITS commands."""
    try:
        config = load_config()
        
        # Initialize ITS config from saved settings or defaults
        its_config = ITSConfig(
            enabled=config.get("reasoning_modes", {}).get("its", {}).get("enabled", False),
            depth_mode=config.get("reasoning_modes", {}).get("its", {}).get("depth_mode", "deep"),
            debug=True,  # Always show detailed output as discussed
            cache_timeout=config.get("reasoning_modes", {}).get("its", {}).get("cache_timeout", 300),
            api_key=load_api_key(),  # Load API key from config
            model_name=get_model_api_name()  # Load model name from config
        )
        
        # Initialize processor and command handler
        processor = ITSProcessor(its_config)
        handler = ITSCommandHandler(processor, its_config)
        
        # Initialize state
        state.its_enabled = its_config.enabled
        state.its_depth_mode = its_config.depth_mode
        state.its_processor = processor if its_config.enabled else None
        
        # Handle the command
        if not submenu_choice:
            return f"""
🧠 Iterative Thought Sculpting
════════════════════════
Status: {'Enabled ✅' if its_config.enabled else 'Disabled ❌'}
Cache: {'Active' if processor.cache.is_valid() else 'Empty'}

Available commands:
1. Toggle ITS (currently {'on' if its_config.enabled else 'off'})
2. Clear reasoning cache
3. Back to main menu

Enter your choice (1-3): """
        
        # Handle menu choices
        if submenu_choice == "1":  # Toggle ITS
            its_config.enabled = not its_config.enabled
            state.its_enabled = its_config.enabled
            state.its_processor = processor if its_config.enabled else None
            
            # Save to config
            if "reasoning_modes" not in config:
                config["reasoning_modes"] = {}
            if "its" not in config["reasoning_modes"]:
                config["reasoning_modes"]["its"] = {}
            config["reasoning_modes"]["its"]["enabled"] = its_config.enabled
            save_config(config)
            
            return f"✅ ITS mode {'enabled' if its_config.enabled else 'disabled'}.\n\nPress Enter to return to ITS menu."
            
        elif submenu_choice == "2":  # Clear cache
            processor.cache.reset()
            return "✅ Reasoning cache cleared.\n\nPress Enter to return to ITS menu."
            
        elif submenu_choice == "3":  # Back to main menu
            return ""
            
        return "Invalid choice. Press Enter to return to ITS menu."
        
    except Exception as e:
        return f"❌ Error handling ITS command: {str(e)}\n\nPress Enter to return to ITS menu."