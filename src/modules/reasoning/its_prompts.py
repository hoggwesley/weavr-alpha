from typing import Dict, List, Any, Tuple
import json

class ITSPrompts:
    """Manages prompts and Gemini API integration for ITS"""
    
    @staticmethod
    def query_analysis_prompt(query: str) -> str:
        """Generate prompt for step 1: Query Analysis"""
        return f"""You are performing step 1 of 4 in the Iterative Thought Sculpting process: Query Analysis.
        
Your task is to break down this query into clear, logical reasoning steps:
"{query}"

Requirements:
1. Identify 3-5 key steps needed to fully answer this query
2. Each step should build on previous steps
3. Steps should be clear and actionable
4. Consider what knowledge or context might be needed

Output Format (JSON):
{{
    "steps": [
        {{
            "step": "Step description",
            "reasoning": "Why this step is necessary",
            "knowledge_needed": ["Type of knowledge needed for this step"]
        }}
    ]
}}

Analyze the query and provide your response in the specified JSON format."""

    @staticmethod
    def knowledge_injection_prompt(query: str, steps: List[Dict], knowledge: Dict) -> str:
        """Generate prompt for step 2: Knowledge Injection"""
        return f"""You are performing step 2 of 4 in the Iterative Thought Sculpting process: Knowledge Injection.

Query: "{query}"

Previously identified steps:
{json.dumps(steps, indent=2)}

Available Knowledge:
{json.dumps(knowledge, indent=2)}

Your task is to expand each reasoning step using the provided knowledge.

Requirements:
1. For each step, incorporate relevant knowledge
2. Identify any gaps where knowledge is missing
3. Note any conflicts or uncertainties
4. Maintain logical flow between steps

Output Format (JSON):
{{
    "expanded_steps": [
        {{
            "original_step": "Step from previous analysis",
            "expanded_reasoning": "Detailed reasoning with knowledge incorporated",
            "knowledge_used": ["References to specific knowledge used"],
            "gaps_identified": ["Any missing information needed"]
        }}
    ]
}}

Analyze the steps with the knowledge provided and respond in the specified JSON format."""

    @staticmethod
    def self_critique_prompt(query: str, expanded_steps: Dict) -> str:
        """Generate prompt for step 3: Self-Critique"""
        return f"""You are performing step 3 of 4 in the Iterative Thought Sculpting process: Self-Critique.

Query: "{query}"

Expanded reasoning steps:
{json.dumps(expanded_steps, indent=2)}

Your task is to critically evaluate the reasoning process so far.

Requirements:
1. Identify any logical gaps or inconsistencies
2. Point out potential biases or assumptions
3. Suggest specific improvements
4. Consider alternative perspectives
5. Evaluate the strength of evidence/knowledge used

Output Format (JSON):
{{
    "critique": {{
        "logical_gaps": ["Identified gaps in reasoning"],
        "assumptions": ["Underlying assumptions made"],
        "improvements": ["Specific suggestions for improvement"],
        "alternatives": ["Alternative perspectives to consider"]
    }}
}}

Critically evaluate the reasoning and respond in the specified JSON format."""

    @staticmethod
    def final_synthesis_prompt(
        query: str,
        expanded_steps: Dict,
        critique: Dict,
        knowledge: Dict
    ) -> str:
        """Generate prompt for step 4: Final Synthesis"""
        return f"""You are performing step 4 of 4 in the Iterative Thought Sculpting process: Final Synthesis.

Query: "{query}"

Expanded reasoning:
{json.dumps(expanded_steps, indent=2)}

Self-critique:
{json.dumps(critique, indent=2)}

Available Knowledge:
{json.dumps(knowledge, indent=2)}

Your task is to synthesize a final, comprehensive response.

Requirements:
1. Address the original query directly
2. Incorporate insights from all previous steps
3. Address identified gaps and critiques
4. Present a clear, well-structured response
5. Maintain appropriate scope and depth

Synthesize all the information and provide a clear, comprehensive response.
Do not include JSON formatting in your response.
Write in a clear, engaging style while maintaining accuracy and depth.""" 