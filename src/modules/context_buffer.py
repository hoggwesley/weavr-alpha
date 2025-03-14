"""
Module for managing conversation context in Weavr AI.
"""
import re
from typing import List, Dict

class ContextBuffer:
    def __init__(self, max_exchanges=10):
        self.exchanges: List[Dict[str, str]] = []
        self.max_exchanges = max_exchanges

    def add_exchange(self, user_message: str, ai_response: str):
        """Add a new exchange to the buffer."""
        self.exchanges.append({
            "user": user_message,
            "ai": ai_response
        })
        
        # Keep only the last max_exchanges
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges:]

    def get_formatted_context(self, current_query: str = "") -> str:
        """Get formatted conversation history."""
        if not self.exchanges:
            return ""
            
        context = "Previous conversation:\n"
        for exchange in self.exchanges[-5:]:  # Only include last 5 exchanges
            context += f"User: {exchange['user']}\nAI: {exchange['ai']}\n\n"
        return context.strip()

    def clear(self):
        """Clear all conversation history."""
        self.exchanges = []

    def filter_by_relevance(self, query: str, threshold: float = 0.3):
        """
        Filter conversation history to keep only relevant exchanges.
        Basic implementation using keyword matching.
        """
        if not query or not self.exchanges:
            return
            
        # Extract keywords from query (simple implementation)
        query_words = set(re.findall(r'\w+', query.lower()))
        
        # Keep exchanges that share keywords with the query
        filtered = []
        for exchange in self.exchanges:
            exchange_text = f"{exchange['user']} {exchange['ai']}".lower()
            exchange_words = set(re.findall(r'\w+', exchange_text))
            
            # Calculate simple relevance score
            common_words = query_words.intersection(exchange_words)
            relevance = len(common_words) / len(query_words) if query_words else 0
            
            if relevance >= threshold:
                filtered.append(exchange)
                
        self.exchanges = filtered

# Global context buffer instance
context_buffer = ContextBuffer()