from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ITSCommandResult:
    """Result of an ITS command execution"""
    message: str
    success: bool = True
    should_continue: bool = True  # False if we should exit command processing

class ITSCommandHandler:
    """Handles ITS-specific commands"""
    
    def __init__(self, its_processor, config):
        self.processor = its_processor
        self.config = config
        
    def handle_command(self, command: str) -> ITSCommandResult:
        """Process ITS-related commands"""
        parts = command.lower().split()
        
        if len(parts) < 2:
            return ITSCommandResult(
                "Invalid ITS command. Available commands:\n"
                "/its on|off - Toggle ITS mode\n"
                "/its depth deep|shallow - Set reasoning depth\n"
                "/its reset - Clear reasoning cache\n"
                "/continue refining - Continue with previous reasoning",
                success=False
            )
            
        cmd = parts[1]
        
        handlers = {
            "on": self._handle_on,
            "off": self._handle_off,
            "depth": self._handle_depth,
            "reset": self._handle_reset
        }
        
        handler = handlers.get(cmd)
        if not handler:
            return ITSCommandResult(
                f"Unknown ITS command: {cmd}",
                success=False
            )
            
        return handler(parts[2:] if len(parts) > 2 else [])
    
    def _handle_on(self, args: list) -> ITSCommandResult:
        """Handle /its on"""
        if self.config.enabled:
            return ITSCommandResult("ITS is already enabled.")
            
        self.config.enabled = True
        return ITSCommandResult(
            "Iterative Thought Sculpting mode enabled. Each query will now be processed "
            "using multi-step reasoning with detailed output."
        )
    
    def _handle_off(self, args: list) -> ITSCommandResult:
        """Handle /its off"""
        if not self.config.enabled:
            return ITSCommandResult("ITS is already disabled.")
            
        self.config.enabled = False
        self.processor.cache.reset()  # Clear cache when disabling
        return ITSCommandResult("Iterative Thought Sculpting mode disabled.")
    
    def _handle_depth(self, args: list) -> ITSCommandResult:
        """Handle /its depth [deep|shallow]"""
        if not args:
            return ITSCommandResult(
                "Please specify depth mode: deep or shallow",
                success=False
            )
            
        mode = args[0].lower()
        if mode not in ["deep", "shallow"]:
            return ITSCommandResult(
                "Invalid depth mode. Use 'deep' or 'shallow'.",
                success=False
            )
            
        self.config.depth_mode = mode
        self.processor.cache.reset()  # Clear cache when changing depth
        
        msg = (
            f"ITS depth set to {mode} mode.\n"
            f"{'Using all 4 reasoning steps.' if mode == 'deep' else 'Using simplified 2-step reasoning.'}"
        )
        return ITSCommandResult(msg)
    
    def _handle_reset(self, args: list) -> ITSCommandResult:
        """Handle /its reset"""
        self.processor.cache.reset()
        return ITSCommandResult("ITS reasoning cache cleared.")
    
    def handle_continue_refining(self) -> Tuple[bool, str]:
        """Handle /continue refining command"""
        can_continue, message = self.processor.cache.handle_continue_refining()
        return can_continue, message 