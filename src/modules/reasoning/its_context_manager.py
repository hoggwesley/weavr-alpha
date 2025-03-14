from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class TokenLimits:
    """Token limits for each ITS step"""
    step_1: int = 1000  # Query analysis
    step_2: int = 2000  # Knowledge injection
    step_3: int = 1500  # Self-critique
    step_4: int = 2000  # Final synthesis

class ITSContextManager:
    """Manages context windows and token usage for ITS steps"""
    
    def __init__(self, token_limits: TokenLimits, debugger=None):
        self.token_limits = token_limits
        self.debugger = debugger
        self.logger = logging.getLogger(__name__)
        
        # Reserve some tokens for system prompts and overhead
        self.system_overhead = 200
        
    def get_token_limit(self, step: int) -> int:
        """Get token limit for a specific step"""
        limit_map = {
            1: self.token_limits.step_1,
            2: self.token_limits.step_2,
            3: self.token_limits.step_3,
            4: self.token_limits.step_4
        }
        return limit_map.get(step, 1000) - self.system_overhead
    
    def prune_chat_history(self, history: List[Dict[str, Any]], step: int) -> List[Dict[str, Any]]:
        """Dynamically prune chat history based on step requirements"""
        token_limit = self.get_token_limit(step)
        estimated_tokens = self._estimate_tokens(str(history))
        
        if estimated_tokens <= token_limit:
            return history
            
        # If we need to prune, keep most recent exchanges
        pruned_history = []
        current_tokens = 0
        
        for msg in reversed(history):
            msg_tokens = self._estimate_tokens(str(msg))
            if current_tokens + msg_tokens > token_limit * 0.7:  # Leave 30% for new content
                break
            pruned_history.insert(0, msg)
            current_tokens += msg_tokens
        
        if self.debugger:
            self.debugger.log_pruning(
                f"Pruned chat history from {len(history)} to {len(pruned_history)} messages"
            )
        
        return pruned_history
    
    def optimize_knowledge_context(self, 
                                 knowledge: Dict[str, Any], 
                                 step: int,
                                 query: str) -> Dict[str, Any]:
        """Filter knowledge based on step requirements and relevance"""
        if step != 2:  # Only inject knowledge in step 2
            return {}
            
        token_limit = self.get_token_limit(step)
        estimated_tokens = self._estimate_tokens(str(knowledge))
        
        if estimated_tokens <= token_limit:
            return knowledge
            
        # If we need to prune, prioritize most relevant sections
        pruned_knowledge = self._prioritize_knowledge(knowledge, query, token_limit)
        
        if self.debugger:
            self.debugger.log_pruning(
                f"Pruned knowledge context to fit within {token_limit} tokens"
            )
        
        return pruned_knowledge
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def _prioritize_knowledge(self, 
                            knowledge: Dict[str, Any], 
                            query: str,
                            token_limit: int) -> Dict[str, Any]:
        """Prioritize knowledge sections based on relevance to query"""
        # This is a simplified version - in practice, you'd want more sophisticated
        # relevance scoring and filtering
        pruned = {}
        current_tokens = 0
        
        # Sort knowledge items by estimated relevance
        # For now, just take the first few that fit within token limit
        for key, value in knowledge.items():
            item_tokens = self._estimate_tokens(str(value))
            if current_tokens + item_tokens > token_limit * 0.8:  # Leave 20% for overhead
                break
            pruned[key] = value
            current_tokens += item_tokens
        
        return pruned 