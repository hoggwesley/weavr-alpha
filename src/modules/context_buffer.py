"""
Module for managing conversation context in Weavr AI.
"""
import re
from typing import List, Dict

class ContextBuffer:
    def __init__(self, max_exchanges=10, max_context_chars=4000):
        self.exchanges: List[Dict[str, str]] = []
        self.max_exchanges = max_exchanges
        self.max_context_chars = max_context_chars  # Limit total context size

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
        """
        Get formatted conversation history, optimized for context window management.
        Prioritizes recent and relevant exchanges while staying within size limits.
        """
        if not self.exchanges:
            return ""
        
        # First, filter by relevance to the current query
        if current_query:
            self.filter_by_relevance(current_query)
            
        # Start with most recent exchanges
        context_parts = []
        total_chars = 0
        
        # Always include the most recent exchange
        if self.exchanges:
            most_recent = self.exchanges[-1]
            recent_text = f"User: {most_recent['user']}\nAI: {most_recent['ai']}\n\n"
            context_parts.append(recent_text)
            total_chars += len(recent_text)
        
        # Add previous exchanges, respecting the character limit
        for exchange in reversed(self.exchanges[:-1]):
            exchange_text = f"User: {exchange['user']}\nAI: {exchange['ai']}\n\n"
            if total_chars + len(exchange_text) <= self.max_context_chars:
                context_parts.insert(0, exchange_text)  # Insert at beginning to maintain chronological order
                total_chars += len(exchange_text)
            else:
                # If we can't fit the full exchange, add a summary instead
                summary = f"[Previous exchange about: {exchange['user'][:50]}...]\n\n"
                if total_chars + len(summary) <= self.max_context_chars:
                    context_parts.insert(0, summary)
                    total_chars += len(summary)
                else:
                    break  # Can't fit any more content
        
        if context_parts:
            return "Previous conversation:\n" + "".join(context_parts).strip()
        return ""

    def clear(self):
        """Clear all conversation history."""
        self.exchanges = []

    def filter_by_relevance(self, query: str, threshold: float = 0.2):
        """
        Filter conversation history to keep only relevant exchanges.
        Uses improved keyword matching with special handling for document titles.
        """
        if not query or not self.exchanges:
            return
            
        # Extract keywords from query
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Always keep the most recent exchange
        most_recent = self.exchanges[-1] if self.exchanges else None
        
        # Filter older exchanges by relevance
        filtered = []
        if most_recent:
            filtered.append(most_recent)
            
        for exchange in self.exchanges[:-1]:  # Skip the most recent one we already added
            exchange_text = f"{exchange['user']} {exchange['ai']}".lower()
            
            # Check for document title mentions (special case for knowledge base queries)
            doc_title_match = False
            doc_title_pattern = r'document:\s*([^\.]+)'
            doc_titles = re.findall(doc_title_pattern, exchange_text)
            
            for title in doc_titles:
                if any(title_word in query.lower() for title_word in title.lower().split()):
                    doc_title_match = True
                    break
            
            # Calculate relevance score
            exchange_words = set(re.findall(r'\b\w+\b', exchange_text))
            common_words = query_words.intersection(exchange_words)
            
            # Calculate relevance score with higher weight for title matches
            relevance = len(common_words) / len(query_words) if query_words else 0
            if doc_title_match:
                relevance += 0.3  # Boost for document title matches
                
            if relevance >= threshold:
                filtered.append(exchange)
                
        self.exchanges = filtered

# Global context buffer instance
context_buffer = ContextBuffer()