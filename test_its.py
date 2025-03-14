import asyncio
import yaml
import logging
from src.modules.reasoning.its_processor import ITSProcessor, ITSConfig
from src.modules.reasoning.its_context_manager import TokenLimits

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_its_processor():
    # Load configuration from config.yaml
    with open('config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    
    # Create ITS configuration
    token_limits = TokenLimits(
        step_1=config_data['reasoning_modes']['its']['token_limits']['step_1'],
        step_2=config_data['reasoning_modes']['its']['token_limits']['step_2'],
        step_3=config_data['reasoning_modes']['its']['token_limits']['step_3'],
        step_4=config_data['reasoning_modes']['its']['token_limits']['step_4']
    )
    
    its_config = ITSConfig(
        enabled=config_data['reasoning_modes']['its']['enabled'],
        depth_mode="shallow",  # Use shallow mode for faster testing
        debug=config_data['reasoning_modes']['its']['debug'],
        token_limits=token_limits,
        cache_timeout=config_data['reasoning_modes']['its']['cache_timeout'],
        model_name=config_data['gemini']['model_name'],
        api_key=config_data['gemini']['api_key']
    )
    
    # Create ITS processor
    processor = ITSProcessor(its_config)
    
    # Test query
    query = "What is 2+2?"  # Simple query for testing
    knowledge_base = {"example": "Basic arithmetic operations."}
    chat_history = [{"role": "user", "content": "I need help with math."}]
    
    print(f"Processing query: {query}")
    try:
        # Set a timeout for the process call
        result = await asyncio.wait_for(
            processor.process(query, knowledge_base, chat_history),
            timeout=30  # 30 seconds timeout
        )
        print(f"Result: {result}")
    except asyncio.TimeoutError:
        print("Operation timed out after 30 seconds")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_its_processor()) 