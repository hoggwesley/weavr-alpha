from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import aiohttp
from google import genai
import re

from .its_cache import ITSCache
from .its_debugger import ITSDebugger
from .its_context_manager import ITSContextManager, TokenLimits

@dataclass
class ITSConfig:
    """Configuration for ITS processing"""
    enabled: bool = False
    depth_mode: str = "deep"  # "deep" or "shallow"
    debug: bool = False
    token_limits: TokenLimits = field(default_factory=TokenLimits)
    cache_timeout: int = 300
    model_name: str = ""
    api_key: str = ""

class ITSProcessor:
    """Main processor for Iterative Thought Sculpting"""
    
    def __init__(self, config: ITSConfig):
        self.config = config
        self.cache = ITSCache(cache_timeout=config.cache_timeout)
        self.debugger = ITSDebugger(debug_enabled=config.debug)
        self.context_manager = ITSContextManager(
            token_limits=config.token_limits,
            debugger=self.debugger
        )
        self.logger = logging.getLogger(__name__)
        
    async def process(self, 
                     query: str,
                     knowledge_base: Any,
                     chat_history: List[Dict[str, Any]]) -> str:
        """Process a query using ITS reasoning"""
        try:
            if self.config.depth_mode == "shallow":
                return await self._process_shallow(query, knowledge_base, chat_history)
            return await self._process_deep(query, knowledge_base, chat_history)
        except Exception as e:
            self.logger.error(f"Error in ITS processing: {str(e)}", exc_info=True)
            return f"Error during reasoning process: {str(e)}"
            
    async def _process_shallow(self,
                             query: str,
                             knowledge_base: Any,
                             chat_history: List[Dict[str, Any]]) -> str:
        """Shallow processing - only steps 1 and 4"""
        # Step 1: Query Analysis
        reasoning_steps = await self._step_query_analysis(query, chat_history)
        
        # Create expanded steps with knowledge
        expanded_steps = []
        pruned_knowledge = self.context_manager.optimize_knowledge_context(
            knowledge_base, 2, query
        )
        
        for step in reasoning_steps:
            expanded_step = {
                "step": step,
                "knowledge_applied": pruned_knowledge
            }
            expanded_steps.append(expanded_step)
        
        # Skip critique in shallow mode
        critique = {"gaps": [], "improvements": []}
        
        # Step 4: Final Synthesis
        final_response = await self._step_final_synthesis(
            expanded_steps, critique, chat_history
        )
        
        return final_response
        
    async def _process_deep(self,
                          query: str,
                          knowledge_base: Any,
                          chat_history: List[Dict[str, Any]]) -> str:
        """Deep processing - all four steps"""
        # Step 1: Query Analysis
        self.debugger.log_step_start(1)
        reasoning_steps = await self._step_query_analysis(query, chat_history)
        
        # Step 2: Knowledge Injection
        self.debugger.log_step_start(2)
        expanded_steps = []
        for step in reasoning_steps:
            pruned_knowledge = self.context_manager.optimize_knowledge_context(
                knowledge_base, 2, step
            )
            expanded_step = await self._step_knowledge_injection(
                step, pruned_knowledge, chat_history
            )
            expanded_steps.append(expanded_step)
        
        # Step 3: Self-Critique
        self.debugger.log_step_start(3)
        critique = await self._step_self_critique(
            expanded_steps, chat_history
        )
        
        # Step 4: Final Synthesis
        self.debugger.log_step_start(4)
        final_response = await self._step_final_synthesis(
            expanded_steps, critique, chat_history
        )
        
        return final_response
    
    async def _step_query_analysis(self,
                                 query: str,
                                 chat_history: List[Dict[str, Any]]) -> List[str]:
        """Step 1: Break down query into reasoning steps"""
        api_input = f"Break down the following question into a structured reasoning process. Return a list of 4-6 clear steps: {query}"
        response = await self.gemini_api_call(api_input)
        
        # Extract steps from the response
        response_text = response.get("response", "")
        
        # Parse the response to extract steps
        steps = []
        for line in response_text.split('\n'):
            # Look for lines that start with numbers, asterisks, or other common list markers
            if re.match(r'^\s*(\d+\.|[\*\-•]|\(\d+\))\s+', line):
                # Extract the step text (remove the marker)
                step_text = re.sub(r'^\s*(\d+\.|[\*\-•]|\(\d+\))\s+', '', line).strip()
                if step_text:
                    steps.append(step_text)
        
        # If no steps were found, try to extract sections with headers
        if not steps:
            # Look for sections like "1. Step One" or "Step 1:" or "**Step 1:**"
            step_sections = re.findall(r'(?:^|\n)(?:\d+\.\s+|\*\*)?(?:Step\s+\d+:?|\d+\.|\*\*|\d+\))\s*([^\n]+)', response_text)
            steps = [s.strip() for s in step_sections if s.strip()]
        
        # If still no steps, use default steps
        if not steps:
            steps = [
                "Define key terms and context",
                "Analyze core components",
                "Consider implications",
                "Synthesize findings"
            ]
            
        # Limit to a reasonable number of steps
        if len(steps) > 8:
            steps = steps[:8]
        elif len(steps) < 3:
            # Add default steps if too few were found
            default_steps = [
                "Define key terms and context",
                "Analyze core components",
                "Consider implications",
                "Synthesize findings"
            ]
            steps.extend(default_steps[len(steps):4])
        
        self.cache.update_step(1, steps, 500)
        return steps
    
    async def _step_knowledge_injection(self,
                                      step: str,
                                      knowledge: Dict[str, Any],
                                      chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 2: Inject knowledge into reasoning steps"""
        api_input = f"Step: {step}\nRelevant knowledge: {knowledge}\nConsider prior context: {chat_history}"
        response = await self.gemini_api_call(api_input)
        expanded = response.get("response", {
            "step": step,
            "knowledge_applied": "Example knowledge application"
        })
        self.cache.update_step(2, expanded, 800)
        return expanded
    
    async def _step_self_critique(self,
                                expanded_steps: List[Dict[str, Any]],
                                chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 3: Review and critique reasoning"""
        api_input = f"Review the following response for inconsistencies and suggest refinements: {expanded_steps}\nConsider prior conversation context: {chat_history}"
        response = await self.gemini_api_call(api_input)
        critique = response.get("response", {
            "gaps": ["Example gap"],
            "improvements": ["Example improvement"]
        })
        self.cache.update_step(3, critique, 600)
        return critique
    
    async def _step_final_synthesis(self,
                                  expanded_steps: List[Dict[str, Any]],
                                  critique: Dict[str, Any],
                                  chat_history: List[Dict[str, Any]]) -> str:
        """Step 4: Synthesize final response"""
        api_input = f"Using the refined reasoning and critique, synthesize a clear, structured final answer: {expanded_steps}\nCritique Considerations: {critique}"
        response = await self.gemini_api_call(api_input)
        final_response = response.get("response", "Example final response incorporating all steps")
        self.cache.update_step(4, final_response, 900)
        return final_response
        
    def continue_refinement(self) -> str:
        """Continue refining previous reasoning"""
        if not self.cache.is_valid():
            return "No valid reasoning cache found to refine."
            
        # TODO: Implement refinement logic using cached results
        return "Refinement not yet implemented"

    async def gemini_api_call(self, api_input: str) -> Any:
        """Make an API call to the Gemini model"""
        try:
            # Configure the Gemini API client
            client = genai.Client(api_key=self.config.api_key)
            
            # Log the API input for debugging
            self.logger.debug(f"API Input: {api_input}")
            
            # Generate content
            response = client.models.generate_content(
                model=self.config.model_name,
                contents=api_input
            )
            
            # Log the API response for debugging
            self.logger.debug(f"API Response: {response.text}")
            
            # Return the response
            return {"response": response.text}
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}", exc_info=True)
            return {"response": f"Error in API call: {str(e)}"} 