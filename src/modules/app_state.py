"""
Global state management for Weavr AI.
"""

class AppState:
    def __init__(self):
        self.use_knowledge = False
        self.cot_enabled = False
        self.structured_mem = None
        self.debug_mode = False
        self.its_enabled = False
        self.its_depth_mode = "deep"  # Can be "deep" or "shallow"
        self.its_processor = None

# Global state instance
state = AppState() 