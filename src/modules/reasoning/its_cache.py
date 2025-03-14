from typing import Dict, Optional, Tuple, Any
from time import time

class ITSCache:
    """Manages caching for Iterative Thought Sculpting reasoning steps"""
    
    def __init__(self, cache_timeout: int = 300):
        self.step_results: Dict[str, Any] = {}
        self.last_query: Optional[str] = None
        self.last_update_time: Optional[float] = None
        self.token_usage = {
            'step1': 0,
            'step2': 0,
            'step3': 0,
            'step4': 0
        }
        self.cache_timeout = cache_timeout
    
    def is_valid(self) -> bool:
        """Check if cache is valid (not expired and has content)"""
        if not self.last_update_time or not self.last_query:
            return False
        
        time_elapsed = time() - self.last_update_time
        return time_elapsed < self.cache_timeout and bool(self.step_results)
    
    def handle_continue_refining(self) -> Tuple[bool, str]:
        """Returns (can_continue, message)"""
        if not self.is_valid():
            return (False, "There is no cached reasoning to refine. Would you like to start a new ITS session? (yes/no)")
        return (True, "")
    
    def update_step(self, step: int, result: Any, tokens_used: int):
        """Update cache with new step result"""
        self.step_results[f'step{step}'] = result
        self.token_usage[f'step{step}'] = tokens_used
        self.last_update_time = time()
    
    def get_step_result(self, step: int) -> Optional[Any]:
        """Get result for a specific step"""
        return self.step_results.get(f'step{step}')
    
    def reset(self):
        """Clear all cached data"""
        self.__init__(self.cache_timeout)
    
    def start_new_session(self, query: str):
        """Start a new reasoning session"""
        self.reset()
        self.last_query = query 