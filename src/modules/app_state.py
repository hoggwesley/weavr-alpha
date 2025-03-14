"""
Global state management for Weavr AI.
"""

class AppState:
    def __init__(self):
        self.use_knowledge = False
        self.cot_enabled = False
        self.structured_mem = None
        self.debug_mode = False

# Global state instance
state = AppState() 