"""
Module for enhancing queries to improve RAG retrieval accuracy.
"""
import re
from typing import List, Dict, Tuple

class QueryEnhancer:
    """Enhances queries to improve retrieval performance."""
    
    def __init__(self):
        # Keywords for different query types
        self.code_keywords = ['function', 'class', 'method', 'implementation', 
                             'define', 'declared', 'code', 'syntax', 'variable', 
                             'parameter', 'return', 'import']
        
        self.file_keywords = ['file', 'script', 'module', '.py', '.md', '.txt', 
                             '.json', '.yaml', '.yml', 'document']
        
        self.directory_keywords = ['directory', 'folder', 'structure', 
                                  'organization', 'path', 'location']
        
        self.model_keywords = ['model', 'ai', 'mixtral', 'mistral', 'llm', 
                              'generate', 'output', 'predict', 'inference']
    
    def detect_query_type(self, query: str) -> Dict[str, bool]:
        """Detect the type of query based on keywords."""
        query_lower = query.lower()
        
        return {
            "is_code_query": any(kw in query_lower for kw in self.code_keywords),
            "is_file_query": any(kw in query_lower for kw in self.file_keywords),
            "is_directory_query": any(kw in query_lower for kw in self.directory_keywords),
            "is_model_query": any(kw in query_lower for kw in self.model_keywords)
        }
    
    def extract_file_references(self, query: str) -> List[str]:
        """Extract potential file references from the query."""
        # Look for common file patterns
        file_pattern = r'\b[\w\-\.]+\.(py|md|txt|json|yaml|yml)\b'
        files = re.findall(file_pattern, query.lower())
        
        # Look for words that might be filenames (without extension)
        words = query.split()
        potential_files = [w for w in words if '_' in w and len(w) > 3]
        
        return list(set(files + potential_files))
    
    def enhance_query(self, query: str) -> str:
        """Enhance a query with additional relevant terms."""
        query_types = self.detect_query_type(query)
        query_terms = [query]  # Start with the original query
        
        # Add terms based on query type
        if query_types["is_code_query"]:
            query_terms.append("function class method implementation code")
            
        if query_types["is_file_query"]:
            files = self.extract_file_references(query)
            if files:
                query_terms.extend([f"filename:{f}" for f in files])
            query_terms.append("file document content")
            
        if query_types["is_directory_query"]:
            query_terms.append("directory structure organization files")
            
        if query_types["is_model_query"]:
            query_lower = query.lower()
            if 'mixtral' in query_lower:
                query_terms.append("mixtral_8x7b_v01 Mixtral-8x7B-Instruct-v0.1")
            elif 'mistral' in query_lower:
                query_terms.append("mistral_7b_v01 Mistral-7B-Instruct-v0.1")
            else:
                query_terms.append("ai model generate response")
        
        # Join all terms
        enhanced_query = " ".join(query_terms)
        return enhanced_query

# Create a singleton instance
query_enhancer = QueryEnhancer()

def enhance_query(query: str) -> str:
    """Enhance a query for better retrieval."""
    return query_enhancer.enhance_query(query)
