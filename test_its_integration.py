"""
Test script for the ITS system integration.
"""
import asyncio
import sys
import os

# Add the src directory to the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "src"))

from modules.reasoning import ITSConfig, ITSProcessor
from modules.config_loader import load_api_key, get_model_api_name

async def test_its():
    """Test the ITS system."""
    print("Testing ITS system...")
    
    # Initialize ITS config
    its_config = ITSConfig(
        enabled=True,
        depth_mode="deep",
        debug=True,
        cache_timeout=300,
        api_key=load_api_key(),
        model_name=get_model_api_name()
    )
    
    # Initialize processor
    processor = ITSProcessor(its_config)
    
    # Test query
    query = "What is 2+2?"
    print(f"Processing query: {query}")
    
    # Process query
    result = await processor.process(query, [])
    
    print("\nResult:")
    print(result)
    
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(test_its()) 