import logging
from typing import Dict

class ITSDebugger:
    """Handles debug output and token tracking for ITS"""
    
    STEP_DESCRIPTIONS = {
        1: "Query Analysis",
        2: "Knowledge Injection",
        3: "Self-Critique",
        4: "Final Synthesis"
    }
    
    def __init__(self, debug_enabled: bool = False):
        self.debug_enabled = debug_enabled
        self.logger = logging.getLogger(__name__)
        self.total_tokens_used = 0
    
    def log_step_start(self, step: int):
        """Log the start of a reasoning step"""
        if not self.debug_enabled:
            return
            
        description = self.STEP_DESCRIPTIONS.get(step, "Unknown Step")
        print(f"\n[ITS] Starting Step {step}/4: {description}")
    
    def log_step_complete(self, step: int, tokens_used: int, token_limit: int):
        """Log step completion with token usage"""
        if not self.debug_enabled:
            return
            
        description = self.STEP_DESCRIPTIONS.get(step, "Unknown Step")
        self.total_tokens_used += tokens_used
        
        # Check if we're approaching token limits
        usage_percent = (tokens_used / token_limit) * 100
        if usage_percent > 90:
            print(f"[ITS] ⚠️ Step {step}/4: {description} (Tokens Used: {tokens_used} / {token_limit})")
            print(f"[ITS] ⚠️ High token usage detected ({usage_percent:.1f}% of limit)")
        else:
            print(f"[ITS] Step {step}/4: {description} (Tokens Used: {tokens_used} / {token_limit})")
    
    def log_pruning(self, reason: str):
        """Log when content is being pruned"""
        if self.debug_enabled:
            print(f"[ITS] ⚠️ {reason}")
    
    def log_cache_operation(self, operation: str):
        """Log cache-related operations"""
        if self.debug_enabled:
            print(f"[ITS] Cache: {operation}")
    
    def get_token_summary(self) -> Dict[str, int]:
        """Get summary of token usage"""
        return {
            'total_tokens': self.total_tokens_used
        }
    
    def reset_token_count(self):
        """Reset token counting for new session"""
        self.total_tokens_used = 0 