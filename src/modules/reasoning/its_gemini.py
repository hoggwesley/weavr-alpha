from typing import Dict, Any, Tuple
import json
import logging
from google.generativeai import GenerativeModel
from .its_prompts import ITSPrompts

class ITSGemini:
    """Handles Gemini API interactions for ITS"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-001"):
        self.model = GenerativeModel(model_name)
        self.prompts = ITSPrompts()
        self.logger = logging.getLogger(__name__)
        
    async def _call_gemini(self, prompt: str) -> Tuple[Any, int]:
        """Make a call to Gemini API and return response with token count"""
        try:
            response = await self.model.generate_content_async(prompt)
            # TODO: Get actual token count from response
            estimated_tokens = len(prompt.split()) + len(response.text.split())
            
            return response.text, estimated_tokens
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}", exc_info=True)
            raise
            
    async def query_analysis(self, query: str) -> Tuple[Dict, int]:
        """Execute step 1: Query Analysis"""
        prompt = self.prompts.query_analysis_prompt(query)
        response, tokens = await self._call_gemini(prompt)
        
        try:
            result = json.loads(response)
            return result, tokens
        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON from query analysis response")
            raise
            
    async def knowledge_injection(self,
                                query: str,
                                steps: Dict,
                                knowledge: Dict) -> Tuple[Dict, int]:
        """Execute step 2: Knowledge Injection"""
        prompt = self.prompts.knowledge_injection_prompt(query, steps, knowledge)
        response, tokens = await self._call_gemini(prompt)
        
        try:
            result = json.loads(response)
            return result, tokens
        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON from knowledge injection response")
            raise
            
    async def self_critique(self,
                          query: str,
                          expanded_steps: Dict) -> Tuple[Dict, int]:
        """Execute step 3: Self-Critique"""
        prompt = self.prompts.self_critique_prompt(query, expanded_steps)
        response, tokens = await self._call_gemini(prompt)
        
        try:
            result = json.loads(response)
            return result, tokens
        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON from self-critique response")
            raise
            
    async def final_synthesis(self,
                            query: str,
                            expanded_steps: Dict,
                            critique: Dict,
                            knowledge: Dict) -> Tuple[str, int]:
        """Execute step 4: Final Synthesis"""
        prompt = self.prompts.final_synthesis_prompt(
            query, expanded_steps, critique, knowledge
        )
        response, tokens = await self._call_gemini(prompt)
        return response, tokens  # No JSON parsing needed for final response 